import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
import optax
jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------
# 1. Differentiable Black-Scholes IV Inversion
# ---------------------------------------------------------
def bs_call_price(S, K, T, r, q, vol):
    d1 = (jnp.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * jnp.sqrt(T))
    d2 = d1 - vol * jnp.sqrt(T)
    return S * jnp.exp(-q * T) * norm.cdf(d1) - K * jnp.exp(-r * T) * norm.cdf(d2)

def bs_vega(S, K, T, r, q, vol):
    d1 = (jnp.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * jnp.sqrt(T))
    return S * jnp.exp(-q * T) * jnp.sqrt(T) * norm.pdf(d1)

def newton_iv(price, S, K, T, r, q, max_iter=100, tol=1e-5):
    vol_init = jnp.float64(0.20)
    
    def cond_fun(state):
        i, _, diff = state
        return jnp.logical_and(i < max_iter, jnp.abs(diff) > tol)

    def body_fun(state):
        i, vol, _ = state
        c_calc = bs_call_price(S, K, T, r, q, vol)
        diff = c_calc - price
        
        vega = bs_vega(S, K, T, r, q, vol)
        vega_safe = jnp.where(vega > 1e-8, vega, 1e-8)
        
        vol_next = vol - diff / vega_safe
        vol_next = jnp.clip(vol_next, 1e-4, 5.0)
        return (i + 1, vol_next, diff)

    init_diff = bs_call_price(S, K, T, r, q, vol_init) - price
    final_state = jax.lax.while_loop(cond_fun, body_fun, (0, vol_init, init_diff))
    return final_state[1]

@jax.custom_vjp
def implied_volatility(price, S, K, T, r, q):
    return newton_iv(price, S, K, T, r, q)

def iv_fwd(price, S, K, T, r, q):
    vol = implied_volatility(price, S, K, T, r, q)
    return vol, (vol, S, K, T, r, q)

def iv_bwd(res, g):
    vol, S, K, T, r, q = res
    vega = bs_vega(S, K, T, r, q, vol)
    vega_safe = jnp.where(vega > 1e-8, vega, 1e-8)
    return (g / vega_safe, None, None, None, None, None)

implied_volatility.defvjp(iv_fwd, iv_bwd)  # type: ignore

# ---------------------------------------------------------
# 2. Heston Characteristic Function
# ---------------------------------------------------------
def heston_char_func(u, T, r, q, kappa, theta, sigma, rho, v0):
    alpha = -0.5 * (u**2 + u * 1j)
    beta = kappa - rho * sigma * u * 1j
    gamma = 0.5 * sigma**2
    
    d = jnp.sqrt(beta**2 - 4 * alpha * gamma)
    r_plus = (beta + d) / (2 * gamma)
    r_minus = (beta - d) / (2 * gamma)
    g = r_minus / r_plus
    
    C = kappa * (r_minus * T - (2 / sigma**2) * jnp.log((1 - g * jnp.exp(-d * T)) / (1 - g)))
    D = r_minus * (1 - jnp.exp(-d * T)) / (1 - g * jnp.exp(-d * T))
    
    return jnp.exp(C * theta + D * v0 + 1j * u * (r - q) * T)

# ---------------------------------------------------------
# 3. Lewis (2001) Fourier Inversion (Fixed-Grid GPU Native)
# ---------------------------------------------------------
def price_heston_lewis(S, K, T, r, q, kappa, theta, sigma, rho, v0):
    # Fixed integration grid - ultra fast on GPUs, no while-loops!
    u_max = 100.0
    N_steps = 1000
    u = jnp.linspace(1e-4, u_max, N_steps)
    du = u[1] - u[0]
    
    z = u - 0.5j
    phi = heston_char_func(z, T, r, q, kappa, theta, sigma, rho, v0)
    
    numerator = phi * jnp.exp(1j * u * jnp.log(S / K))
    denominator = u**2 + 0.25
    integrand_values = jnp.real(numerator / denominator)
    
    integral = jnp.trapezoid(integrand_values, dx=du)
    return S * jnp.exp(-q * T) - (jnp.sqrt(S * K * jnp.exp(-(r + q) * T)) / jnp.pi) * integral

# ---------------------------------------------------------
# 4. Objective Function & Reparameterization
# ---------------------------------------------------------
def transform_heston_params(raw_params):
    return jnp.array([
        jax.nn.softplus(raw_params[0]),             
        jax.nn.softplus(raw_params[1]),             
        jax.nn.softplus(raw_params[2]),             
        jnp.tanh(raw_params[3]),                    
        jax.nn.softplus(raw_params[4])              
    ])

@jax.jit
def heston_calibration_loss(raw_params, S, K_array, T_array, r, q, market_iv_array):
    params = transform_heston_params(raw_params)
    kappa, theta, sigma, rho, v0 = params

    def single_iv(K, T, market_iv):
        price = price_heston_lewis(S, K, T, r, q, kappa, theta, sigma, rho, v0)
        model_iv = implied_volatility(price, S, K, T, r, q)
        return (model_iv - market_iv)**2

    mse_array = jax.vmap(single_iv)(K_array, T_array, market_iv_array)
    return jnp.sqrt(jnp.mean(mse_array))

# ---------------------------------------------------------
# 5. Optax Training Loop
# ---------------------------------------------------------
def calibrate_heston_fourier_ad(market_snapshot, steps=500, lr=0.01):
    S = market_snapshot['spot']
    r = market_snapshot['r']
    q = market_snapshot.get('q', 0.0)
    df = market_snapshot['options']
    
    K_array = jnp.array(df['K'].values)
    T_array = jnp.array(df['T'].values)
    market_iv_array = jnp.array(df['IV'].values)

    def inv_softplus(x): return jnp.log(jnp.exp(x) - 1.0)
    def inv_tanh(x): return jnp.arctanh(x)

    init_raw = jnp.array([
        inv_softplus(2.0), inv_softplus(0.04), inv_softplus(0.3), inv_tanh(-0.7), inv_softplus(0.04)
    ])

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(init_raw)

    @jax.jit
    def optimization_step(raw_params, state):
        loss_val, grads = jax.value_and_grad(heston_calibration_loss)(raw_params, S, K_array, T_array, r, q, market_iv_array)
        updates, new_state = optimizer.update(grads, state, raw_params)
        new_params = optax.apply_updates(raw_params, updates)
        return new_params, new_state, loss_val

    raw_params = init_raw
    print("Starting Heston Fourier/AD Calibration (Fixed Grid)...")
    for i in range(steps):
        raw_params, opt_state, loss_val = optimization_step(raw_params, opt_state)
        if i % 100 == 0:
            print(f"Step {i} | RMSE: {loss_val:.6f}")
    
    final_params = transform_heston_params(raw_params)
    
    return {
        'kappa': float(final_params[0]),
        'theta': float(final_params[1]),
        'sigma': float(final_params[2]),
        'rho':   float(final_params[3]),
        'v0':    float(final_params[4])
    }
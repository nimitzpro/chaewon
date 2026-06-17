import jax
import jax.numpy as jnp
# import QuantLib as ql

# # dS(t) = r S(t) dt + sqrt(V(t)) S(t) (rho dW_1(t) + sqrt(1 - rho**2) dW_2(t))
# # dV(t) = kappa(theta - V(t))dt + sigma sqrt(V(t)) dW_2(t)

def simulate_single_path(params, key, n_steps, T):
    """Internal function to simulate one path using lax.scan"""
    dt = T / n_steps
    
    # Extract market data and model parameters
    S0 = params['S0']
    r = params.get('r', 0.0)  # Your FRED rate will flow in here
    q = params.get('q', 0.0)  # Dividend yield
    
    v0 = params['v0']
    kappa = params['kappa']
    theta = params['theta']
    sigma = params['sigma']
    rho = params['rho']
    
    log_S0 = jnp.log(S0)
    
    # Draw all noise for this single path upfront
    dW = jax.random.normal(key, shape=(n_steps, 2)) * jnp.sqrt(dt)
    
    def step(carry, dW_t):
        log_S, V = carry
        V_plus = jnp.maximum(V, 0.0) # Full truncation scheme for stability
        
        dW1, dW2 = dW_t[0], dW_t[1]
        dWS = rho * dW2 + jnp.sqrt(1.0 - rho**2) * dW1
        
        # Euler-Maruyama on log price
        log_S_new = log_S + (r - q - 0.5 * V_plus) * dt + jnp.sqrt(V_plus) * dWS
        V_new = V + kappa * (theta - V_plus) * dt + sigma * jnp.sqrt(V_plus) * dW2
        
        return (log_S_new, V_new), (log_S_new, V_new)
        
    _, (log_S_path, V_path) = jax.lax.scan(step, (log_S0, v0), dW)
    
    # Prepend the initial state to get n_steps + 1
    log_S_full = jnp.concatenate([jnp.array([log_S0]), log_S_path])
    V_full = jnp.concatenate([jnp.array([v0]), V_path])
    
    # Stack into shape (n_steps+1, 2)
    return jnp.stack([log_S_full, V_full], axis=-1)

def simulate_heston(params: dict, key: jax.Array, n_paths: int, n_steps: int, T: float) -> jnp.ndarray:
    """
    The unified interface expected by the diagnostics layer.
    Returns: jnp.ndarray of shape (n_paths, n_steps+1, 2)
    """
    # Split the PRNG key for n_paths
    keys = jax.random.split(key, n_paths)
    
    # Vectorize the single path simulator across all keys simultaneously
    paths = jax.vmap(lambda k: simulate_single_path(params, k, n_steps, T))(keys)
    
    return paths


# # QuantLib analytic Heston pricing
# today = ql.Date().todaysDate()
# riskFreeTS = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.05, ql.Actual365Fixed()))
# dividendTS = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.01, ql.Actual365Fixed()))
# initialValue = ql.QuoteHandle(ql.SimpleQuote(100))

# v0, kappa, theta, sigma, rho = 0.005, 0.8, 0.008, 0.1, 0.2
# hestonProcess = ql.HestonProcess(riskFreeTS, dividendTS, initialValue, v0, kappa, theta, sigma, rho)
# hestonModel = ql.HestonModel(hestonProcess)
# engine = ql.AnalyticHestonEngine(hestonModel)
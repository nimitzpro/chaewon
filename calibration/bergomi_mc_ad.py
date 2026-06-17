# pyrefly: ignore [missing-import]
import jax
import jax.numpy as jnp
import optax
from functools import partial

from models.bergomi import simulate_bergomi
from .fourier_ad import implied_volatility

def transform_bergomi_params(raw_params):
    """Maps unconstrained optimizer guesses to valid Bergomi boundaries."""
    return {
        'eta': jax.nn.softplus(raw_params[0]),
        'rho': jnp.tanh(raw_params[1]),
        # H <= 0.5. For PD/1F models, H MUST be allowed to go negative!
        'H':   0.5 - jax.nn.softplus(raw_params[2]), 
        'v0':  jax.nn.softplus(raw_params[3])
    }

@partial(jax.jit, static_argnums=(7,))
def bergomi_mc_pricer(params, key, S, K, T, r, q, model_type):
    """Prices a European Call via batched Monte Carlo simulation."""
    params = {**params, 'S0': S, 'r': r, 'q': q}
    # 4096 paths ensures the MC noise is small enough for stable gradients
    paths = simulate_bergomi(params, key, n_paths=4096, n_steps=100, T=T, model_type=model_type)
    
    terminal_S = jnp.exp(paths[:, -1, 0])
    payoffs = jnp.maximum(terminal_S - K, 0.0)
    return jnp.exp(-r * T) * jnp.mean(payoffs)

@partial(jax.jit, static_argnums=(8,))
def bergomi_calibration_loss(raw_params, static_key, S, K_array, T_array, r, q, market_iv_array, model_type):
    params = transform_bergomi_params(raw_params)
    
    def single_iv(K, T, market_iv):
        # CRITICAL: We pass the EXACT SAME static_key every time. 
        # Freezing the random noise ensures the loss landscape doesn't jitter during optimization.
        price = bergomi_mc_pricer(params, static_key, S, K, T, r, q, model_type)
        model_iv = implied_volatility(price, S, K, T, r, q)
        return (model_iv - market_iv)**2
        
    mse_array = jax.vmap(single_iv)(K_array, T_array, market_iv_array)
    return jnp.sqrt(jnp.mean(mse_array))

def calibrate_bergomi_mc_ad(market_snapshot, static_key, model_type="rough", steps=200, lr=0.02):
    """Optax training loop for MC-AD Bergomi calibration."""
    S = market_snapshot['spot']
    r = market_snapshot['r']
    q = market_snapshot.get('q', 0.0)
    df = market_snapshot['options']
    
    K_array = jnp.array(df['K'].values)
    T_array = jnp.array(df['T'].values)
    market_iv_array = jnp.array(df['IV'].values)

    def inv_softplus(x): return jnp.log(jnp.exp(x) - 1.0)
    def inv_tanh(x): return jnp.arctanh(x)

    # Initial guesses: eta=1.5, rho=-0.7, H=0.1, v0=0.04
    init_raw = jnp.array([
        inv_softplus(1.5), 
        inv_tanh(-0.7), 
        inv_softplus(0.5 - 0.1), 
        inv_softplus(0.04)
    ])

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(init_raw)

    @jax.jit
    def optimization_step(raw_params, state):
        loss_val, grads = jax.value_and_grad(bergomi_calibration_loss)(
            raw_params, static_key, S, K_array, T_array, r, q, market_iv_array, model_type
        )
        updates, new_state = optimizer.update(grads, state, raw_params)
        new_params = optax.apply_updates(raw_params, updates)
        return new_params, new_state, loss_val

    raw_params = init_raw
    print(f"Starting {model_type} Bergomi MC-AD Calibration...")
    for i in range(steps):
        raw_params, opt_state, loss_val = optimization_step(raw_params, opt_state)
        if i % 20 == 0:
            print(f"Step {i} | RMSE: {loss_val:.6f}")
    
    final_params_dict = transform_bergomi_params(raw_params)
    
    # Cast to float for standard python dictionary output
    return {k: float(v) for k, v in final_params_dict.items()}
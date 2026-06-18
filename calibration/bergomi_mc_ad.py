# pyrefly: ignore [missing-import]
import jax
import jax.numpy as jnp
import optax
from functools import partial

from models.bergomi import simulate_bergomi
from .fourier_ad import implied_volatility

def transform_bergomi_params(raw_params, model_type):
    """Maps unconstrained optimizer guesses to valid Bergomi boundaries based on model type."""
    
    if model_type == "rough":
        # Forces H strictly into (0, 0.5]
        H_val = 0.5 * jax.nn.sigmoid(raw_params[2])
    else:
        # Allows H to go negative (bounded above by 0.5)
        H_val = 0.5 - jax.nn.softplus(raw_params[2])
        
    return {
        'eta': jax.nn.softplus(raw_params[0]),
        'rho': jnp.tanh(raw_params[1]),
        'H':   H_val, 
        'v0':  jax.nn.softplus(raw_params[3])
    }

@partial(jax.jit, static_argnums=(7,))
def bergomi_mc_pricer(params, key, S, K, T, r, q, model_type):
    # 0. RE-INJECT the market parameters into the dictionary for the simulator!
    sim_params = {**params, 'S0': S, 'r': r, 'q': q}
    
    # Pass sim_params instead of params
    paths = simulate_bergomi(sim_params, key, n_paths=4096, n_steps=100, T=T, model_type=model_type)
    terminal_S = jnp.exp(paths[:, -1, 0])
    
    # 1. Identify if the Call is Out-of-the-Money (K >= S)
    is_otm_call = K >= S
    
    # 2. Price strictly Out-of-the-Money (Calls for K >= S, Puts for K < S)
    payoffs = jnp.where(is_otm_call, 
                        jnp.maximum(terminal_S - K, 0.0),  # OTM Call Payoff
                        jnp.maximum(K - terminal_S, 0.0))  # OTM Put Payoff
    
    price_otm = jnp.exp(-r * T) * jnp.mean(payoffs)
    
    # 3. Convert Puts back to Calls via Put-Call Parity: C = P + S*e^-qT - K*e^-rT
    call_price = jnp.where(is_otm_call, 
                           price_otm, 
                           price_otm + S * jnp.exp(-q * T) - K * jnp.exp(-r * T))
    
    return call_price

@partial(jax.jit, static_argnums=(8,))
def bergomi_calibration_loss(raw_params, static_key, S, K_array, T_array, r, q, market_iv_array, model_type):
    # Pass the model_type down to the transformer
    params = transform_bergomi_params(raw_params, model_type)
    
    def single_iv(K, T, market_iv):
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
    
    final_params_dict = transform_bergomi_params(raw_params, model_type)
    
    # Cast to float for standard python dictionary output
    return {k: float(v) for k, v in final_params_dict.items()}
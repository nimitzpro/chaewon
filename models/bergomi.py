import jax
import jax.numpy as jnp

def simulate_single_path_bergomi(params, key, n_steps, T, model_type="rough"):
    dt = T / n_steps
    
    S0 = params['S0']
    v0 = params['v0']  # Flat forward variance curve assumption
    r = params.get('r', 0.0)
    q = params.get('q', 0.0)
    
    eta = params['eta']
    rho = params['rho']
    H = params['H']
    # epsilon is the time-shift used for Path-Dependent/1-Factor models. 
    # Abi Jaber & Li fix this at 1/52 (approx 1 week).
    epsilon = params.get('epsilon', 1/52) 
    
    # 1. Generate Noise
    dW = jax.random.normal(key, shape=(n_steps, 2)) * jnp.sqrt(dt)
    dW1, dW2 = dW[:, 0], dW[:, 1]
    
    dWS = rho * dW2 + jnp.sqrt(1.0 - rho**2) * dW1
    dWV = dW2
    
    # 2. Construct the Volterra Kernel Matrix (Lower Triangular)
    row_indices = jnp.arange(n_steps)[:, None]
    col_indices = jnp.arange(n_steps)[None, :]
    delta_t = (row_indices - col_indices + 1.0) * dt
    mask = row_indices >= col_indices
    
    # --- THE MODEL REGISTRY SWITCH ---
    if model_type == "rough":
        # Singularity at t=0. Prevent delta_t <= 0 where mask is False from causing NaN gradients.
        safe_delta_t = jnp.where(mask, delta_t, 1.0)
        kernel = jnp.where(mask, safe_delta_t**(H - 0.5), 0.0)
    elif model_type == "path_dependent":
        # Smooths the singularity. Prevent negative delta_t + epsilon from causing NaN gradients.
        safe_base = jnp.where(mask, delta_t + epsilon, 1.0)
        kernel = jnp.where(mask, safe_base**(H - 0.5), 0.0)
    elif model_type == "one_factor":
        # The Markovian proxy using exponential decay.
        safe_delta_t = jnp.where(mask, delta_t, 0.0)
        kernel = jnp.where(mask, (epsilon**(H - 0.5)) * jnp.exp(-(0.5 - H) * safe_delta_t / epsilon), 0.0)
    else:
        raise ValueError("model_type must be 'rough', 'path_dependent', or 'one_factor'")
        
    # 3. Compute fractional process X_t and exact discrete variance compensator
    # Matrix multiply computes the entire integral history in one GPU operation
    X_path = eta * jnp.dot(kernel, dWV)
    
    # The compensator guarantees E[V_t] = v0, preventing martingale leakage
    var_compensator = 0.5 * (eta**2) * jnp.sum(kernel**2 * dt, axis=1)
    
    # 4. Compute Spot Variance V_t
    V_path = v0 * jnp.exp(X_path - var_compensator)
    V_shifted = jnp.concatenate([jnp.array([v0]), V_path[:-1]]) # Align to start of interval
    
    # 5. Integrate Stock Price S_t
    drift = (r - q - 0.5 * V_shifted) * dt
    diffusion = jnp.sqrt(V_shifted) * dWS
    log_S_path = jnp.log(S0) + jnp.cumsum(drift + diffusion)
    
    log_S_full = jnp.concatenate([jnp.array([jnp.log(S0)]), log_S_path])
    V_full = jnp.concatenate([jnp.array([v0]), V_path])
    
    return jnp.stack([log_S_full, V_full], axis=-1)

def simulate_bergomi(params, key, n_paths, n_steps, T, model_type="rough"):
    keys = jax.random.split(key, n_paths)
    return jax.vmap(lambda k: simulate_single_path_bergomi(params, k, n_steps, T, model_type))(keys)
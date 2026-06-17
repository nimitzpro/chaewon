import jax
import jax.numpy as jnp
import sigkerax

def apply_lead_lag(paths):
    """
    Transforms a tensor of shape (n_paths, n_steps, n_features)
    into a lead-lag tensor of shape (n_paths, 2 * n_steps - 1, 2 * n_features).
    Necessary for the signature kernel to capture pathwise quadratic variation.
    """
    n_paths, n_steps, n_feats = paths.shape
    
    # Repeat steps to create the interleaved timeline
    lead = jnp.repeat(paths, 2, axis=1)[:, 1:, :]
    lag = jnp.repeat(paths, 2, axis=1)[:, :-1, :]
    
    return jnp.concatenate([lead, lag], axis=-1)

def compute_sig_mmd(real_paths, sim_paths, static_kernel_fn):
    """
    Computes the unbiased MMD^2 estimator between real market trajectories 
    and calibrated model simulations using the sigkerax kernel.
    """
    # 1. Apply lead-lag transformation
    real_ll = apply_lead_lag(real_paths)
    sim_ll = apply_lead_lag(sim_paths)
    
    # 2. Compute the three cross-kernel matrices using sigkerax
    # Squeeze the trailing dimension (1) returned by sigkerax
    K_xx = static_kernel_fn(real_ll, real_ll).squeeze(-1)
    K_xy = static_kernel_fn(real_ll, sim_ll).squeeze(-1)
    K_yy = static_kernel_fn(sim_ll, sim_ll).squeeze(-1)
    
    # 3. Compute unbiased means (removing diagonals for self-similarities)
    m = K_xx.shape[0]
    n = K_yy.shape[0]
    
    mean_K_xx = (jnp.sum(K_xx) - jnp.trace(K_xx)) / (m * (m - 1))
    mean_K_yy = (jnp.sum(K_yy) - jnp.trace(K_yy)) / (n * (n - 1))
    mean_K_xy = jnp.mean(K_xy)
    
    # 4. Return the scalar MMD^2 distance
    return mean_K_xx - 2.0 * mean_K_xy + mean_K_yy
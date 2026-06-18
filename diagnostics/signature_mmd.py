# import jax
# import jax.numpy as jnp

# def normalize_and_bound_paths(paths):
#     """
#     Anchors paths to zero and strictly bounds their maximum magnitude
#     into a [-1.0, 1.0] hypercube to prevent PDE solver overflow.
#     """
#     # 1. Anchor to 0.0
#     anchored = paths - paths[:, 0:1, :]
    
#     # 2. Strict MaxAbs Bounding per feature
#     max_vals = jnp.max(jnp.abs(anchored), axis=(0, 1), keepdims=True)
#     safe_max = jnp.where(max_vals > 0, max_vals, 1.0)
    
#     # Compress into unit hypercube
#     bounded_paths = anchored / safe_max
    
#     return bounded_paths

# def apply_lead_lag(paths):
#     lead = jnp.repeat(paths, 2, axis=1)[:, 1:, :]
#     lag = jnp.repeat(paths, 2, axis=1)[:, :-1, :]
#     return jnp.concatenate([lead, lag], axis=-1)

# def compute_sig_mmd(real_paths, sim_paths, signature_kernel):
#     # 1. Bound and Transform
#     real_ll = apply_lead_lag(normalize_and_bound_paths(real_paths))
#     sim_ll = apply_lead_lag(normalize_and_bound_paths(sim_paths))
    
#     # 2. Compute exact Kernel Matrices using the PDE solver
#     # .squeeze() handles any trailing dimensions left by the solver
#     K_xx = signature_kernel.kernel_matrix(real_ll, real_ll).squeeze()
#     K_yy = signature_kernel.kernel_matrix(sim_ll, sim_ll).squeeze()
#     K_xy = signature_kernel.kernel_matrix(real_ll, sim_ll).squeeze()
    
#     # 3. Biased MMD Estimator
#     mean_K_xx = jnp.mean(K_xx)
#     mean_K_yy = jnp.mean(K_yy)
#     mean_K_xy = jnp.mean(K_xy)
    
#     mmd_squared = mean_K_xx - 2.0 * mean_K_xy + mean_K_yy
    
#     # 4. Underflow Protection
#     return jnp.maximum(mmd_squared, 0.0)

import jax
import jax.numpy as jnp

def apply_lead_lag(paths):
    """
    Transforms a tensor of shape (n_paths, n_steps, n_features)
    into a lead-lag tensor of shape (n_paths, 2 * n_steps - 1, 2 * n_features).
    """
    lead = jnp.repeat(paths, 2, axis=1)[:, 1:, :]
    lag = jnp.repeat(paths, 2, axis=1)[:, :-1, :]
    return jnp.concatenate([lead, lag], axis=-1)

def compute_sig_mmd(real_paths, sim_paths, static_kernel_fn):
    """
    Computes the BIASED MMD^2 estimator using the exact PDE signature kernel.
    Raw paths are used to prevent exponential blowup in the PDE solver.
    """
    # 1. Apply lead-lag transformation
    real_ll = apply_lead_lag(real_paths)
    sim_ll = apply_lead_lag(sim_paths)
    
    # 2. Compute the cross-kernel matrices
    K_xx = static_kernel_fn(real_ll, real_ll).squeeze(-1)
    K_xy = static_kernel_fn(real_ll, sim_ll).squeeze(-1)
    K_yy = static_kernel_fn(sim_ll, sim_ll).squeeze(-1)
    
    # 3. Compute BIASED means (Retains the diagonal to guarantee >= 0)
    mean_K_xx = jnp.mean(K_xx)
    mean_K_yy = jnp.mean(K_yy)
    mean_K_xy = jnp.mean(K_xy)
    
    # 4. Return the strictly non-negative distance
    mmd_squared = mean_K_xx - 2.0 * mean_K_xy + mean_K_yy
    
    return jnp.maximum(mmd_squared, 0.0)


# import jax
# import jax.numpy as jnp

@jax.jit
def apply_lead_lag(paths):
    """
    Transforms a tensor of shape (n_paths, n_steps, n_features)
    into a lead-lag tensor of shape (n_paths, 2 * n_steps - 1, 2 * n_features).
    """
    lead = jnp.repeat(paths, 2, axis=1)[:, 1:, :]
    lag = jnp.repeat(paths, 2, axis=1)[:, :-1, :]
    return jnp.concatenate([lead, lag], axis=-1)

def compute_sig_mmd_batched(real_paths, sim_paths, static_kernel_fn, batch_size=256):
    """
    Computes the strictly positive, biased MMD^2 estimator using the exact PDE signature kernel.
    Batches the cross-kernel evaluations to prevent XLA memory explosions,
    allowing full utilization of the simulation paths.
    """
    # 1. Apply lead-lag transformation
    real_ll = apply_lead_lag(real_paths)
    sim_ll = apply_lead_lag(sim_paths)
    
    n_paths = real_ll.shape[0]
    
    # Accumulators
    sum_K_xx = 0.0
    sum_K_yy = 0.0
    sum_K_xy = 0.0
    
    # Block-matrix iteration loop
    for i in range(0, n_paths, batch_size):
        end_i = min(i + batch_size, n_paths)
        X_chunk_i = real_ll[i:end_i]
        Y_chunk_i = sim_ll[i:end_i]
        
        for j in range(0, n_paths, batch_size):
            end_j = min(j + batch_size, n_paths)
            X_chunk_j = real_ll[j:end_j]
            Y_chunk_j = sim_ll[j:end_j]
            
            # Compute parallelized sub-blocks using the PDE kernel_matrix
            # .squeeze() handles the trailing dimensions from the solver
            K_xx_block = static_kernel_fn(X_chunk_i, X_chunk_j).squeeze()
            K_yy_block = static_kernel_fn(Y_chunk_i, Y_chunk_j).squeeze()
            K_xy_block = static_kernel_fn(X_chunk_i, Y_chunk_j).squeeze()
            
            sum_K_xx += jnp.sum(K_xx_block)
            sum_K_yy += jnp.sum(K_yy_block)
            sum_K_xy += jnp.sum(K_xy_block)
            
    # Compute BIASED means (Retains the diagonal to guarantee distance >= 0)
    total_elements = float(n_paths * n_paths)
    mean_K_xx = sum_K_xx / total_elements
    mean_K_yy = sum_K_yy / total_elements
    mean_K_xy = sum_K_xy / total_elements
    
    mmd_squared = mean_K_xx - 2.0 * mean_K_xy + mean_K_yy
    
    # 4. Floating-Point Underflow Protection
    return jnp.maximum(mmd_squared, 0.0)
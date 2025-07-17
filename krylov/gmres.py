import torch
import numpy as np

def gmres(A, b, x0=None, max_iter=None, rtol=1e-5, M=None, x_true=None, **kwargs):
    """
    Solves the system Ax=b using the GMRES method.
    Handles both torch tensors and numpy arrays.
    """
    n = A.shape[0]
    if max_iter is None:
        max_iter = n * 10

    is_torch = isinstance(b, torch.Tensor)
    if is_torch:
        device, dtype = b.device, b.dtype
        A_matvec = lambda v: A @ v
        x = torch.zeros_like(b) if x0 is None else x0.clone()
        M_solve = M.solve if M is not None else (lambda v: v)
    else: # is numpy
        A_matvec = lambda v: A @ v
        x = np.zeros_like(b) if x0 is None else x0.copy()
        M_solve = M.solve if M is not None else (lambda v: v)

    r = b - A_matvec(x)
    r_norm = np.linalg.norm(r.cpu() if is_torch else r)
    b_norm = np.linalg.norm(b.cpu() if is_torch else b)

    residuals = [r_norm / b_norm]
    errors = []
    if x_true is not None: errors.append(np.linalg.norm((x - x_true).cpu() if is_torch else (x-x_true)))

    V = [r / r_norm]
    H = np.zeros((max_iter + 1, max_iter))
    
    for j in range(max_iter):
        w = A_matvec(M_solve(V[-1]))
        
        for i, v_i in enumerate(V):
            H[i, j] = torch.dot(w, v_i) if is_torch else np.dot(w, v_i)
            w = w - H[i, j] * v_i
        
        H[j + 1, j] = torch.linalg.norm(w) if is_torch else np.linalg.norm(w)

        if H[j + 1, j] < 1e-10:
            break
        
        V.append(w / H[j + 1, j])

        e1 = np.zeros(j + 2)
        e1[0] = 1.0
        
        # Solve the least-squares problem H*y = r_norm * e1
        y, _, _, _ = np.linalg.lstsq(H[:j + 2, :j + 1], r_norm * e1, rcond=None)
        
        # Form the solution update
        update = sum(y[i] * v_i for i, v_i in enumerate(V[:-1]))
        x_new = x + update
        
        res_norm = np.linalg.norm(H[:j+2, :j+1] @ y - r_norm * e1) / b_norm
        residuals.append(res_norm)
        if x_true is not None: errors.append(np.linalg.norm((x_new - x_true).cpu() if is_torch else (x_new-x_true)))

        if res_norm < rtol:
            break

    # Final update after loop
    e1 = np.zeros(len(V))
    e1[0] = 1.0
    y, _, _, _ = np.linalg.lstsq(H[:len(V), :len(V)-1], r_norm * e1, rcond=None)
    update = sum(y[i] * v_i for i, v_i in enumerate(V[:-1]))
    x += update

    return (residuals, errors) if x_true is not None else (residuals, x)
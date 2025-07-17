import torch

def conjugate_gradient(A, b, x0=None, max_iter=None, rtol=1e-5, x_true=None):
    """
    Solves the symmetric positive-definite system Ax=b using the Conjugate Gradient method.
    """
    n = A.shape[0]
    if max_iter is None:
        max_iter = n * 10

    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - A @ x
    p = r.clone()
    rs_old = torch.dot(r, r)
    
    b_norm = torch.linalg.norm(b)
    
    residuals, errors = [], []
    if x_true is not None: errors.append(torch.linalg.norm(x - x_true))
    residuals.append(torch.linalg.norm(r) / b_norm)

    for i in range(max_iter):
        Ap = A @ p
        alpha = rs_old / torch.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        
        rs_new = torch.dot(r, r)
        
        if x_true is not None: errors.append(torch.linalg.norm(x - x_true))
        residuals.append(torch.sqrt(rs_new) / b_norm)
        
        if torch.sqrt(rs_new) < rtol * b_norm:
            break
            
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
        
    return (residuals, errors) if x_true is not None else (residuals, x)

def preconditioned_conjugate_gradient(A, b, M, x0=None, max_iter=None, rtol=1e-5, x_true=None):
    """
    Solves the symmetric positive-definite system Ax=b using the Preconditioned Conjugate Gradient method.
    """
    n = A.shape[0]
    if max_iter is None:
        max_iter = n * 10

    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - A @ x
    
    # M.solve(r) computes M_inv @ r
    z = M.solve(r)
    p = z.clone()
    
    rz_old = torch.dot(r, z)
    b_norm = torch.linalg.norm(b)

    residuals, errors = [], []
    if x_true is not None: errors.append(torch.linalg.norm(x - x_true))
    residuals.append(torch.linalg.norm(r) / b_norm)

    for i in range(max_iter):
        Ap = A @ p
        alpha = rz_old / torch.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        
        if torch.linalg.norm(r) < rtol * b_norm:
            if x_true is not None: errors.append(torch.linalg.norm(x - x_true))
            residuals.append(torch.linalg.norm(r) / b_norm)
            break
        
        z = M.solve(r)
        rz_new = torch.dot(r, z)
        
        p = z + (rz_new / rz_old) * p
        rz_old = rz_new

        if x_true is not None: errors.append(torch.linalg.norm(x - x_true))
        residuals.append(torch.linalg.norm(r) / b_norm)

    return (residuals, errors) if x_true is not None else (residuals, x)
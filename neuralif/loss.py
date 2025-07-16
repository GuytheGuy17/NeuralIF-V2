import warnings
import torch
from apps.data import graph_to_matrix

# Note: We no longer need autograd.Function or the standard checkpoint for the loss
warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')


def frobenius_loss(L, A, sparse=True):
    # This function is unchanged
    if type(L) is tuple:
        U = L[1]
        L = L[0]
    else:
        U = L.T
    if sparse:
        r = L@U - A
        return torch.norm(r)
    else:
        A = A.to_dense().squeeze()
        L = L.to_dense().squeeze()
        U = U.to_dense().squeeze()
        return torch.linalg.norm(L@U - A, ord="fro")


def sketched_loss(L, A, c=None, normalized=False):
    # This function is unchanged
    if type(L) is tuple:
        U = L[1]
        L = L[0]
    else:
        U = L.T
    eps = 1e-8
    z = torch.randn((A.shape[0], 1), device=L.device)
    est = L@(U@z) - A@z
    norm = torch.linalg.vector_norm(est, ord=2)
    if normalized and c is None:
        norm = norm / torch.linalg.vector_norm(A@z, ord=2)
    elif normalized:
        norm = norm / (c + eps)
    return norm


def supervised_loss(L, A, x):
    # This function is unchanged
    if type(L) is tuple:
        U = L[1]
        L = L[0]
    else:
        U = L.T
    if x is None:
        with torch.no_grad():
            b = torch.randn((A.shape[0], 1), device=L.device)
            x = torch.linalg.solve(A.to_dense(), b)
    else:
        b = A@x
    res = L@(U@x) - b
    return torch.linalg.vector_norm(res, ord=2)


def dircet_min_loss(L, A, x):
    # This function is unchanged
    if type(L) is tuple:
        U = L[1]
        L = L[0]
    else:
        U = L.T
    if x is None:
        with torch.no_grad():
            b = torch.randn((A.shape[0], 1), device=L.device)
            x = torch.linalg.solve(A.to_dense(), b)
    else:
        b = A@x
    res = L@(U@x)
    return torch.linalg.vector_norm(res, ord=2)

def pcg_proxy(L_mat, U_mat, A, cg_steps: int = 3, preconditioner_solve_steps: int = 5):
    """
    This function remains unchanged. Its value is computed, but its gradient is not.
    """
    def iterative_sparse_solve(A_sparse, b_vec, iterations):
        x = torch.zeros_like(b_vec)
        r = b_vec - A_sparse @ x
        p = r.clone()
        rs_old = torch.dot(r.squeeze(), r.squeeze())
        for _ in range(iterations):
            Ap = A_sparse @ p
            alpha = rs_old / (torch.dot(p.squeeze(), Ap.squeeze()) + 1e-10)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.dot(r.squeeze(), r.squeeze())
            if torch.sqrt(rs_new) < 1e-8:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

    n = A.shape[0]
    b = torch.randn((n, 1), device=A.device, dtype=A.dtype)
    x = torch.zeros_like(b)
    r = b.clone()
    r0_norm = torch.linalg.vector_norm(r) + 1e-16

    y = iterative_sparse_solve(L_mat, r, iterations=preconditioner_solve_steps)
    z = iterative_sparse_solve(U_mat, y, iterations=preconditioner_solve_steps)
    p = z.clone()
    residuals = []
    rz_old = (r * z).sum()

    for i in range(cg_steps):
        Ap = A @ p
        pAp = (p * Ap).sum()
        if torch.abs(pAp) < 1e-12:
            warnings.warn(f"PCG proxy: Unstable alpha denominator. Stopping early.")
            break
        alpha = rz_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        residuals.append(torch.linalg.vector_norm(r) / r0_norm)
        y = iterative_sparse_solve(L_mat, r, iterations=preconditioner_solve_steps)
        z = iterative_sparse_solve(U_mat, y, iterations=preconditioner_solve_steps)
        rz_new = (r * z).sum()
        if torch.abs(rz_old) < 1e-12:
             warnings.warn(f"PCG proxy: Unstable beta denominator. Stopping early.")
             break
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    if not residuals:
        return torch.tensor(1.0, device=A.device)

    return torch.stack(residuals).mean()

def improved_sketch_with_pcg(
    L, A, num_sketches, normalized, pcg_steps, pcg_weight, use_rademacher
):
    """
    Final corrected version. The pcg_proxy acts as a regularizer
    but its gradient path is detached to save memory.
    """
    if isinstance(L, tuple):
        L_mat, U_mat = L
    else:
        L_mat = L
        U_mat = L_mat.T

    # --- GRADIENT PATH 1: The Sketch Loss (Memory Efficient) ---
    # The gradient will flow through this part of the loss normally.
    # This provides the primary learning signal to the GNN.
    n = A.shape[0]
    losses = []
    for _ in range(num_sketches):
        if use_rademacher:
            z = torch.randint(0,2,(n,1),device=L_mat.device,dtype=L_mat.dtype)*2 - 1
        else:
            z = torch.randn((n,1),device=L_mat.device,dtype=L_mat.dtype)
        
        r = L_mat @ (U_mat @ z) - A @ z

        norm_r = torch.linalg.vector_norm(r,2)
        if normalized:
            denom = torch.linalg.vector_norm(A @ z,2) + 1e-16
            norm_r = norm_r / denom
        losses.append(norm_r)
    sketch_loss = torch.stack(losses).mean()

    # --- GRADIENT PATH 2: The PCG Proxy (Detached) ---
    # We compute the forward pass of the pcg_proxy to get its value,
    # but we immediately detach it from the computation graph by using a no_grad context.
    with torch.no_grad():
        proxy_val = pcg_proxy(L_mat, U_mat, A, pcg_steps)
    
    # The total loss is the sketch loss plus the *value* of the proxy.
    # The gradient of the proxy term with respect to the model parameters is now zero.
    total_loss = sketch_loss + pcg_weight * proxy_val

    return total_loss

def loss(output, data, config=None, **kwargs):
    # load the data
    with torch.no_grad():
        A, b = graph_to_matrix(data)
    
    # Ensure A is float32 for the loss function
    A = A.to(torch.float32)
    
    # compute loss
    if config is None:
        # this is the regular loss used to train NeuralIF
        l = sketched_loss(output, A, normalized=False)
    elif config == "sketched":
        l = sketched_loss(output, A, normalized=kwargs.get("normalized", False))
    elif config == "normalized":
        l = sketched_loss(output, A, kwargs.get("c", None), normalized=True)
    elif config == "supervised":
        l = supervised_loss(output, A, data.s.squeeze())
    elif config == "inverted":
        l = supervised_loss(output, A, None)
    elif config == "frobenius":
        l = frobenius_loss(output, A, sparse=False)
    elif config == 'sketch_pcg':
        l = improved_sketch_with_pcg(
            output,
            A,
            num_sketches=kwargs.get('num_sketches',2),
            normalized=kwargs.get('normalized',False),
            pcg_steps=kwargs.get('pcg_steps',3),
            pcg_weight=kwargs.get('pcg_weight',0.1),
            use_rademacher=kwargs.get('use_rademacher',False)
        )
    else:
        raise ValueError("Invalid loss configuration")
            
    return l
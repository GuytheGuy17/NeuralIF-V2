import warnings
import torch
from torch_sparse import spmm

from apps.data import graph_to_matrix

warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')

# Note: All functions are modified to use torch_sparse.spmm for memory efficiency

def frobenius_loss(L, A, sparse=True):
    if type(L) is tuple:
        U, L = L[1], L[0]
    else:
        U = L.T
    
    if sparse:
        # This operation remains inefficient, but this loss function is not the primary issue.
        # The focus is on the training losses like sketch_pcg.
        r = L@U - A
        return torch.norm(r)
    else:
        A_dense = A.to_dense().squeeze()
        L_dense = L.to_dense().squeeze()
        U_dense = U.to_dense().squeeze()
        return torch.linalg.norm(L_dense @ U_dense - A_dense, ord="fro")


def sketched_loss(L, A, c=None, normalized=False):
    if type(L) is tuple:
        U, L = L[1], L[0]
    else:
        U = L.T
    
    eps = 1e-8
    z = torch.randn((A.shape[0], 1), device=L.device)
    
    # Use spmm for sparse-dense multiplication
    U_z = spmm(U.indices(), U.values(), U.shape[0], U.shape[1], z)
    L_Uz = spmm(L.indices(), L.values(), L.shape[0], L.shape[1], U_z)
    A_z = spmm(A.indices(), A.values(), A.shape[0], A.shape[1], z)
    
    est = L_Uz - A_z
    norm = torch.linalg.vector_norm(est, ord=2)
    
    if normalized:
        denom = torch.linalg.vector_norm(A_z, ord=2) if c is None else (c + eps)
        norm = norm / denom
    
    return norm


def pcg_proxy(L_mat, U_mat, A, cg_steps: int = 2, preconditioner_solve_steps: int = 2):
    """ Differentiable PCG proxy using memory-efficient spmm. """
    
    def iterative_sparse_solve(A_sparse, b_vec, iterations):
        x = torch.zeros_like(b_vec)
        r = b_vec.clone()
        p = r.clone()
        rs_old = torch.dot(r.squeeze(), r.squeeze())

        for _ in range(iterations):
            Ap = spmm(A_sparse.indices(), A_sparse.values(), A_sparse.shape[0], A_sparse.shape[1], p)
            alpha = rs_old / (torch.dot(p.squeeze(), Ap.squeeze()) + 1e-10)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.dot(r.squeeze(), r.squeeze())
            if torch.sqrt(rs_new) < 1e-8: break
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
        Ap = spmm(A.indices(), A.values(), A.shape[0], A.shape[1], p)
        pAp = (p * Ap).sum()
        if torch.abs(pAp) < 1e-12: break
        
        alpha = rz_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        residuals.append(torch.linalg.vector_norm(r) / r0_norm)
        
        y = iterative_sparse_solve(L_mat, r, iterations=preconditioner_solve_steps)
        z = iterative_sparse_solve(U_mat, y, iterations=preconditioner_solve_steps)
        rz_new = (r * z).sum()
        if torch.abs(rz_old) < 1e-12: break
        
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    if not residuals: return torch.tensor(1.0, device=A.device)
    return torch.stack(residuals).mean()


def improved_sketch_with_pcg(L, A, **kwargs):
    """ Combined sketch and PCG proxy loss using spmm. """
    if isinstance(L, tuple):
        L_mat, U_mat = L
    else:
        L_mat, U_mat = L, L.T

    n = A.shape[0]
    losses = []
    for _ in range(kwargs.get('num_sketches', 2)):
        if kwargs.get('use_rademacher', False):
            z = torch.randint(0, 2, (n, 1), device=L_mat.device, dtype=L_mat.dtype) * 2 - 1
        else:
            z = torch.randn((n, 1), device=L_mat.device, dtype=L_mat.dtype)
        
        # Use spmm for all sparse-dense products
        U_z = spmm(U_mat.indices(), U_mat.values(), U_mat.shape[0], U_mat.shape[1], z)
        L_Uz = spmm(L_mat.indices(), L_mat.values(), L_mat.shape[0], L_mat.shape[1], U_z)
        A_z = spmm(A.indices(), A.values(), A.shape[0], A.shape[1], z)
        
        r = L_Uz - A_z
        norm_r = torch.linalg.vector_norm(r, 2)
        
        if kwargs.get('normalized', False):
            denom = torch.linalg.vector_norm(A_z, 2) + 1e-16
            norm_r = norm_r / denom
        losses.append(norm_r)
        
    sketch_loss = torch.stack(losses).mean()
    proxy = pcg_proxy(L_mat, U_mat, A, kwargs.get('pcg_steps', 3))

    return sketch_loss + kwargs.get('pcg_weight', 0.1) * proxy


def loss(output, data, config=None, **kwargs):
    with torch.no_grad():
        A, b = graph_to_matrix(data)
    
    if config == 'sketch_pcg':
        l = improved_sketch_with_pcg(output, A, **kwargs)
    elif config == "sketched":
        l = sketched_loss(output, A, normalized=kwargs.get("normalized", False))
    # Add other loss cases here if needed
    else:
        raise ValueError(f"Loss configuration '{config}' not fully handled in this optimized version.")
        
    return l
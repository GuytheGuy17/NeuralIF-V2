import warnings
import torch

from apps.data import graph_to_matrix


warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')


def frobenius_loss(L, A, sparse=True):
    # * Cholesky decomposition style loss
    
    if type(L) is tuple:
        U = L[1]
        L = L[0]
    else:
        U = L.T
    
    if sparse:
        # Not directly supported in pyotrch:
        # https://github.com/pytorch/pytorch/issues/95169
        # https://github.com/rusty1s/pytorch_sparse/issues/45
        r = L@U - A
        return torch.norm(r)
        
    else:
        A = A.to_dense().squeeze()
        L = L.to_dense().squeeze()
        U = U.to_dense().squeeze()
        
        return torch.linalg.norm(L@U - A, ord="fro")


def sketched_loss(L, A, c=None, normalized=False):
    # both cholesky and LU decomposition
    
    if type(L) is tuple:
        U = L[1]
        L = L[0]
    else:
        U = L.T
    
    eps = 1e-8
    
    z = torch.randn((A.shape[0], 1), device=L.device)
    
    # if normalized:
    # z = z / torch.linalg.norm(z) # z-vector also should have unit length
    
    est = L@(U@z) - A@z
    norm = torch.linalg.vector_norm(est, ord=2) # vector norm
    
    if normalized and c is None:
        norm = norm / torch.linalg.vector_norm(A@z, ord=2)
    elif normalized:
        norm = norm / (c + eps)
    
    return norm


def supervised_loss(L, A, x):
    # Note: Ax = b
    
    if type(L) is tuple:
        U = L[1]
        L = L[0]
    else:
        U = L.T
    
    if x is None:
        # if x is None, we recompute the solution in every time step.
        with torch.no_grad():
            b = torch.randn((A.shape[0], 1), device=L.device)
            x = torch.linalg.solve(A.to_dense(), b)
    else:
        b = A@x
        
    res = L@(U@x) - b
    return torch.linalg.vector_norm(res, ord=2)


def dircet_min_loss(L, A, x):
    # get L and U factors
    if type(L) is tuple:
        U = L[1]
        L = L[0]
    else:
        U = L.T
        
    if x is None:
        # if x is None, we recompute the solution in every time step.
        with torch.no_grad():
            b = torch.randn((A.shape[0], 1), device=L.device)
            x = torch.linalg.solve(A.to_dense(), b)
    else:
        b = A@x
        
    res = L@(U@x)
    return torch.linalg.vector_norm(res, ord=2)

def pcg_proxy(L_mat, U_mat, A, cg_steps: int = 3):
    """
    Perform `cg_steps` of Preconditioned CG on a random RHS, and return
    the mean relative residual over those steps:
        mean_i ||r_i|| / ||r_0||
    """
    n = A.shape[0]
    # random right-hand side b and initial guess x0=0
    b = torch.randn((n, 1), device=A.device, dtype=A.dtype)
    x = torch.zeros_like(b)
    # initial residual
    r = b.clone()
    r0_norm = torch.linalg.vector_norm(r, 2) + 1e-16

    # precondition initial residual: z0 = M^{-1} r0 = (U^{-1} . L^{-1}) r0
    # FIX 1: Use `upper=False` instead of `lower=True`.
    # FIX 2: Assign the result directly to `z`, don't unpack.
    z = torch.linalg.solve_triangular(L_mat, r, upper=False)
    # FIX 1: Use `upper=True` instead of `lower=False`.
    # FIX 2: Assign the result directly to `z`, don't unpack.
    z = torch.linalg.solve_triangular(U_mat, z, upper=True)
    p = z.clone()

    residuals = []
    # --- Start of CG loop ---
    for i in range(cg_steps):
        Ap = A @ p
        # To prevent division by zero for the last step's beta calculation
        rz_old = (r * z).sum()
        
        alpha = rz_old / ((p * Ap).sum() + 1e-16)
        x = x + alpha * p
        
        # Check for early convergence to avoid issues with tiny residuals
        if i < cg_steps - 1:
            r = r - alpha * Ap
        
        # record relative residual
        residuals.append(torch.linalg.vector_norm(r, 2) / r0_norm)

        # precondition for next step
        # FIX 1 & 2 applied here as well
        z = torch.linalg.solve_triangular(L_mat, r, upper=False)
        z = torch.linalg.solve_triangular(U_mat, z, upper=True)
        
        beta = (r * z).sum() / (rz_old + 1e-16)
        p = z + beta * p

    # return average relative residual
    return torch.stack(residuals).mean()


def improved_sketch_with_pcg(
    L,
    A,
    num_sketches: int = 2,
    normalized: bool = False,
    pcg_steps: int = 3,
    pcg_weight: float = 0.1,
    use_rademacher: bool = False
    ):
    """
    Sketch-based loss augmented with a CG proxy averaged over its first iterations:
      - Averages sketch residuals over `num_sketches` sketches
      - Optionally normalizes each sketch residual
      - Adds the mean CG relative-residual over `pcg_steps`, weighted by `pcg_weight`
    """
    # unpack factors
    if isinstance(L, tuple):
        L_mat, U_mat = L
    else:
        L_mat = L
        U_mat = L_mat.T

    # sketch term
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

    # pcg proxy (average over early residuals)
    proxy = pcg_proxy(L_mat, U_mat, A, pcg_steps)

    return sketch_loss + pcg_weight * proxy


def loss(output, data, config=None, **kwargs):
    
    # load the data
    with torch.no_grad():
        A, b = graph_to_matrix(data)
    
    # compute loss
    if config is None:
        # this is the regular loss used to train NeuralIF
        l = sketched_loss(output, A, normalized=False)
    
    elif config == "normalized":
        l = sketched_loss(output, A, kwargs.get("c", None), normalized=True)
    
    elif config == "supervised":
        l = supervised_loss(output, A, data.s.squeeze())
        
    elif config == "inverted":
        l = supervised_loss(output, A, None)
    
    elif config == "combined":
        l = combined_loss(output, A, data.s.squeeze())
        
    elif config == "combined-supervised":
        l = combined_loss(output, A, None)
    
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

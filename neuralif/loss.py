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
    # --- Start of Debug Version ---
    print("\n--- [DEBUG] Entering pcg_proxy ---")

    L_dense = L_mat.to_dense()
    U_dense = U_mat.to_dense()
    
    n = A.shape[0]
    b = torch.randn((n, 1), device=A.device, dtype=A.dtype)
    x = torch.zeros_like(b)
    r = b.clone()
    r0_norm = torch.linalg.vector_norm(r, 2) + 1e-16

    z = torch.linalg.solve_triangular(L_dense, r, upper=False)
    z = torch.linalg.solve_triangular(U_dense, z, upper=True)
    
    # Check for NaNs after initial preconditioning
    if torch.isnan(z).any():
        print("[FATAL] NaN detected in 'z' after initial solve. Exiting.")
        # Returning a large number instead of nan to allow training to continue if needed
        return torch.tensor(1e6, device=A.device)

    p = z.clone()

    residuals = []
    for i in range(cg_steps):
        print(f"\n[DEBUG] CG Step {i+1}")
        Ap = A @ p
        rz_old = (r * z).sum()
        
        pAp = (p * Ap).sum()
        alpha = rz_old / (pAp + 1e-10)  # Avoid division by zero
        
        print(f"[DEBUG] rz_old: {rz_old.item()}, pAp: {pAp.item()}, alpha: {alpha.item()}")
        if torch.isnan(alpha):
            print("[FATAL] alpha is NaN!")
            return torch.tensor(1e6, device=A.device)

        x = x + alpha * p
        
        if i < cg_steps - 1:
            r_new = r - alpha * Ap
        else: # On the last step, r_new is not needed for the next beta
            r_new = r 
        
        residuals.append(torch.linalg.vector_norm(r_new, 2) / r0_norm)
        
        # Check if residual is nan before the next step
        if torch.isnan(residuals[-1]).any():
             print(f"[FATAL] Residual at step {i+1} is NaN!")
             return torch.tensor(1e6, device=A.device)

        r = r_new # Update r
        
        z = torch.linalg.solve_triangular(L_dense, r, upper=False)
        z = torch.linalg.solve_triangular(U_dense, z, upper=True)
        
        rz_new = (r * z).sum()
        beta = rz_new / (rz_old + 1e-16)

        print(f"[DEBUG] rz_new: {rz_new.item()}, beta: {beta.item()}")
        if torch.isnan(beta):
            print("[FATAL] beta is NaN!")
            return torch.tensor(1e6, device=A.device)

        p = z + beta * p

    print("--- [DEBUG] Exiting pcg_proxy successfully ---")
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
    elif config == "sketched":
        l = sketched_loss(output, A, normalized=kwargs.get("normalized", False))
    
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

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
            # Note: solve on CPU to avoid potential GPU memory issues with dense A
            A_dense_cpu = A.to_dense().to('cpu')
            b_cpu = b.to('cpu')
            x_cpu = torch.linalg.solve(A_dense_cpu, b_cpu)
            x = x_cpu.to(L.device)
    else:
        b = A@x
        
    res = L@(U@x) - b
    return torch.linalg.vector_norm(res, ord=2)


def pcg_proxy(L_mat, U_mat, A, cg_steps: int = 2, preconditioner_solve_steps: int = 2):
    """
    A scalable and robust version of the PCG proxy loss.
    This version is corrected to be differentiable w.r.t. L_mat and U_mat for training.
    """
    
    # Inner function to solve Ax=b for a sparse A using a few steps of CG
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

    # Main pcg_proxy logic starts here
    n = A.shape[0]
    b = torch.randn((n, 1), device=A.device, dtype=A.dtype)
    x = torch.zeros_like(b)
    r = b.clone()
    r0_norm = torch.linalg.vector_norm(r) + 1e-16

    # CRITICAL FIX: The preconditioner application MUST track gradients.
    # The `no_grad` block is removed.
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
        
        # CRITICAL FIX: This must also track gradients.
        y = iterative_sparse_solve(L_mat, r, iterations=preconditioner_solve_steps)
        z = iterative_sparse_solve(U_mat, y, iterations=preconditioner_solve_steps)
        
        rz_new = (r * z).sum()
        
        if torch.abs(rz_old) < 1e-12:
            warnings.warn(f"PCG proxy: Unstable beta denominator. Stopping early.")
            break
        
        beta = rz_new / rz_old
        
        # CRITICAL FIX: The new direction `p` must remain in the graph to unroll the optimizer.
        # Removing `.detach()` allows gradients to flow through the CG steps.
        p = z + beta * p

        rz_old = rz_new

    if not residuals:
        return torch.tensor(1.0, device=A.device)

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
    Sketch-based loss augmented with a CG proxy averaged over its first iterations.
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
    """
    Computes the loss with a memory-efficient CPU offloading strategy
    to prevent GPU OOM errors during the backward pass.
    """
    # 1. Store original device and move model output to CPU for calculation
    original_device = output[0].device if isinstance(output, tuple) else output.device
    
    if isinstance(output, tuple):
        output_cpu = (output[0].to('cpu'), output[1].to('cpu'))
    else:
        output_cpu = output.to('cpu')
    
    # 2. Move all data-derived tensors to CPU as well
    with torch.no_grad():
        A, b = graph_to_matrix(data)
        A_cpu = A.to('cpu')
        # Check if solution vector 's' exists in the data object
        s_cpu = data.s.squeeze().to('cpu') if hasattr(data, 's') and data.s is not None else None

    # 3. Compute the selected loss function entirely on the CPU
    #    This avoids creating large intermediate gradient tensors on the GPU.
    if config == 'sketch_pcg':
        l_cpu = improved_sketch_with_pcg(
            output_cpu, A_cpu,
            num_sketches=kwargs.get('num_sketches', 2),
            normalized=kwargs.get('normalized', False),
            pcg_steps=kwargs.get('pcg_steps', 3),
            pcg_weight=kwargs.get('pcg_weight', 0.1),
            use_rademacher=kwargs.get('use_rademacher', False)
        )
    elif config is None or config == "sketched":
        l_cpu = sketched_loss(output_cpu, A_cpu, normalized=kwargs.get("normalized", False))
    elif config == "normalized":
        l_cpu = sketched_loss(output_cpu, A_cpu, c=kwargs.get("c"), normalized=True)
    elif config == "supervised":
        l_cpu = supervised_loss(output_cpu, A_cpu, s_cpu)
    elif config == "inverted":
        l_cpu = supervised_loss(output_cpu, A_cpu, None)
    elif config == "frobenius":
        l_cpu = frobenius_loss(output_cpu, A_cpu, sparse=False)
    else:
        # Your original file listed 'combined' losses which were not defined.
        raise ValueError(f"Invalid or undefined loss configuration: {config}")
    
    # 4. Move the final scalar loss back to the original GPU device.
    #    This makes the fix transparent to the rest of your training script.
    return l_cpu.to(original_device)

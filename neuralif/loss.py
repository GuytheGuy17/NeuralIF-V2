import warnings
import torch
from apps.data import graph_to_matrix

warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')


def frobenius_loss(L, A, sparse=True):
    # This function is now called with CPU tensors
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

# All other helper loss functions (sketched_loss, supervised_loss, pcg_proxy,
# improved_sketch_with_pcg) remain exactly as they were in the "detached gradient"
# version. They will now be called with CPU tensors.

def pcg_proxy(L_mat, U_mat, A, cg_steps: int = 3, preconditioner_solve_steps: int = 5):
    # Unchanged
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

def improved_sketch_with_pcg(
    L, A, num_sketches, normalized, pcg_steps, pcg_weight, use_rademacher
):
    if isinstance(L, tuple):
        L_mat, U_mat = L
    else:
        L_mat = L
        U_mat = L_mat.T

    # This will now be called with CPU tensors, which is fine
    sketch_loss = sketched_loss((L_mat, U_mat), A, normalized=normalized)

    # We will differentiate through pcg_proxy, but it will happen on the CPU
    proxy = pcg_proxy(L_mat, U_mat, A, cg_steps, 5) # 5 is the default solve steps

    return sketch_loss + pcg_weight * proxy


def loss(output, data, config=None, **kwargs):
    # --- THIS IS THE CORE OF THE FIX ---
    # 1. Move model output (which is on GPU) to CPU
    if isinstance(output, tuple):
        output_cpu = (output[0].to('cpu'), output[1].to('cpu'))
    else:
        output_cpu = output.to('cpu')
    
    # 2. Get matrix A and also move it to CPU
    with torch.no_grad():
        A, _ = graph_to_matrix(data)
        A_cpu = A.to('cpu')

    # 3. Compute the loss entirely on the CPU
    if config == 'sketch_pcg':
        l = improved_sketch_with_pcg(
            output_cpu,
            A_cpu,
            num_sketches=kwargs.get('num_sketches',2),
            normalized=kwargs.get('normalized',False),
            pcg_steps=kwargs.get('pcg_steps',3),
            pcg_weight=kwargs.get('pcg_weight',0.1),
            use_rademacher=kwargs.get('use_rademacher',False)
        )
    elif config == "sketched":
        l = sketched_loss(output_cpu, A_cpu, normalized=kwargs.get("normalized", False))
    else:
        raise ValueError(f"Invalid loss configuration: {config}")
    
    # 4. Return the final loss tensor. It is on the CPU.
    #    PyTorch's autograd will handle copying gradients back to the GPU model.
    return l
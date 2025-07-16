import warnings
import torch
from torch.autograd import Function
from apps.data import graph_to_matrix

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


def iterative_sparse_solve(A_sparse, b_vec, iterations):
    # This helper function is unchanged
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

# --- THE DEFINITIVE FIX: A custom autograd function for the preconditioner solve ---
class PreconditionerSolve(Function):
    @staticmethod
    def forward(ctx, L_mat, U_mat, r, preconditioner_solve_steps):
        """
        Forward pass uses the fast, approximate iterative solver.
        """
        with torch.no_grad():
            y = iterative_sparse_solve(L_mat, r, preconditioner_solve_steps)
            z = iterative_sparse_solve(U_mat, y, preconditioner_solve_steps)
        
        # Save inputs needed for the backward pass
        ctx.save_for_backward(L_mat, U_mat, z)
        return z

    @staticmethod
    def backward(ctx, grad_z):
        """
        Backward pass uses the gradient of the exact, memory-efficient triangular solve.
        This avoids differentiating through the iterative solver's loops.
        """
        L_mat, U_mat, z = ctx.saved_tensors
        
        # Re-create the forward operation using a differentiable, exact formulation
        # The gradient of this operation will be used as a proxy.
        with torch.enable_grad():
            L_mat_temp = L_mat.detach().requires_grad_(True)
            U_mat_temp = U_mat.detach().requires_grad_(True)
            z_temp = z.detach().requires_grad_(True)
            
            # To get the original `r`, we re-apply the preconditioner: r = P(z) = L*U*z
            r_temp = L_mat_temp @ (U_mat_temp @ z_temp)
        
        # Calculate the vector-Jacobian product using the chain rule
        # This computes (dL/d(L_mat), dL/d(U_mat), dL/d(r))
        grad_L, grad_U, grad_r = torch.autograd.grad(r_temp, (L_mat_temp, U_mat_temp, z_temp), grad_outputs=grad_z)

        # Return gradients for L_mat, U_mat, r, and None for preconditioner_solve_steps
        return grad_L, grad_U, grad_r, None


def pcg_proxy(L_mat, U_mat, A, cg_steps: int = 3, preconditioner_solve_steps: int = 5):
    """
    This function is now modified to use the custom, memory-safe solver.
    """
    n = A.shape[0]
    b = torch.randn((n, 1), device=A.device, dtype=A.dtype)
    x = torch.zeros_like(b)
    r = b.clone()
    r0_norm = torch.linalg.vector_norm(r) + 1e-16

    # Use our custom autograd function for the preconditioning step
    z = PreconditionerSolve.apply(L_mat, U_mat, r, preconditioner_solve_steps)
    
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
        
        # Use our custom autograd function again for the next step
        z = PreconditionerSolve.apply(L_mat, U_mat, r, preconditioner_solve_steps)
        
        rz_new = (r * z).sum()
        if torch.abs(rz_old) < 1e-12: break
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
    This function now calls the updated pcg_proxy with its memory-safe backward pass.
    """
    if isinstance(L, tuple):
        L_mat, U_mat = L
    else:
        L_mat = L
        U_mat = L_mat.T

    # The sketch loss provides a stable, primary gradient signal
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

    # The pcg_proxy now has a memory-safe backward pass.
    # The number of preconditioner solve steps is passed through here.
    preconditioner_solve_steps = 5 # Or get from kwargs if you add it to the argparser
    proxy = pcg_proxy(L_mat, U_mat, A, pcg_steps, preconditioner_solve_steps)

    return sketch_loss + pcg_weight * proxy


def loss(output, data, config=None, **kwargs):
    # This function is fine as it was, ensuring inputs are float32
    with torch.no_grad():
        A, b = graph_to_matrix(data)
    A = A.to(torch.float32)
    
    # compute loss
    if config == 'sketch_pcg':
        l = improved_sketch_with_pcg(
            output,
            A,
            num_sketches=kwargs.get('num_sketches',2),
            normalized=kwargs.get('normalized',False),
            pcg_steps=kwargs.get('pcg_steps',3),
            pcg_weight=kwargs.get('pcg_weight',0.1),
            use_rademacher=kwargs.get('use_rademacher',False)
        )
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
    else:
        raise ValueError(f"Invalid loss configuration: {config}")
            
    return l
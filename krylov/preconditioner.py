# FILE: krylov/preconditioner.py

import torch
import numpy as np
import scipy.sparse.linalg

from neuralif.utils import torch_sparse_to_scipy, time_function

class Preconditioner:
    """A base class to ensure all preconditioners have a consistent interface."""
    def __init__(self):
        self.time = 0.0
        self.breakdown = False

    @property
    def nnz(self):
        return 0

    def solve(self, b: torch.Tensor) -> torch.Tensor:
        return b # Default is identity (no preconditioning)

class Jacobi(Preconditioner):
    """Jacobi (Diagonal) Preconditioner."""
    def __init__(self, A_torch: torch.Tensor):
        super().__init__()
        self.inv_diag = 1.0 / A_torch.to_dense().diagonal()

    @property
    def nnz(self):
        return len(self.inv_diag)
    
    def solve(self, b: torch.Tensor) -> torch.Tensor:
        return self.inv_diag * b

class ScipyILU(Preconditioner):
    """Wrapper for the SciPy ILU preconditioner."""
    def __init__(self, ilu_obj):
        super().__init__()
        self._prec = ilu_obj

    @property
    def nnz(self):
        return self._prec.L.nnz + self._prec.U.nnz

    def solve(self, b: torch.Tensor) -> torch.Tensor:
        b_np = b.cpu().numpy()
        x_np = self._prec.solve(b_np)
        return torch.from_numpy(x_np).to(b.device)

class Learned(Preconditioner):
    """Wrapper for the learned GNN preconditioner."""
    def __init__(self, model, data):
        super().__init__()
        self._model = model
        self._data = data
        self._computed = False
        self.L_scipy = None
        self.U_scipy = None

    def _compute_preconditioner(self):
        """Runs the model and converts the output factors to a SciPy format."""
        start_time = time_function()
        with torch.no_grad():
            L_torch, U_torch, _ = self._model(self._data)
        
        # Convert torch sparse tensors to scipy sparse matrices for solving
        self.L_scipy = torch_sparse_to_scipy(L_torch.cpu()).tocsc()
        self.U_scipy = torch_sparse_to_scipy(U_torch.cpu()).tocsc()

        self.time = time_function() - start_time
        self._computed = True
    
    @property
    def nnz(self):
        if not self._computed: self._compute_preconditioner()
        return self.L_scipy.nnz + self.U_scipy.nnz

    def solve(self, b: torch.Tensor) -> torch.Tensor:
        """Applies the learned preconditioner M_inv = U_inv * L_inv."""
        if not self._computed: self._compute_preconditioner()
        
        # --- THIS IS THE FIX ---
        # Use SciPy's robust sparse triangular solve instead of torch.triangular_solve
        b_np = b.cpu().numpy()
        y = scipy.sparse.linalg.spsolve_triangular(self.L_scipy, b_np, lower=True)
        x_np = scipy.sparse.linalg.spsolve_triangular(self.U_scipy, y, lower=False)
        return torch.from_numpy(x_np).to(b.device)

def get_preconditioner(data, method: str, model=None) -> Preconditioner:
    """Factory function to create the specified preconditioner."""
    if method == "learned":
        if model is None: raise ValueError("A model must be provided for the 'learned' method.")
        return Learned(model, data)

    # For baselines, create the PyTorch tensor on the specified device
    device = data.x.device
    A_torch = torch.sparse_coo_tensor(
        data.edge_index, data.edge_attr.squeeze(),
        size=(data.num_nodes, data.num_nodes),
        device=device,
        dtype=torch.float64
    ).coalesce()
    
    if method == "baseline":
        return Preconditioner()
    elif method == "jacobi":
        return Jacobi(A_torch)
    elif method == "ilu":
        A_scipy_csc = torch_sparse_to_scipy(A_torch.cpu()).tocsc()
        start_time = time_function()
        try:
            ilu_op = scipy.sparse.linalg.spilu(A_scipy_csc, drop_tol=1e-4, fill_factor=10)
            prec = ScipyILU(ilu_op)
        except Exception as e:
            print(f"\nWARNING: SciPy ILU factorization failed for a sample: {e}")
            prec = Preconditioner()
            prec.breakdown = True
        prec.time = time_function() - start_time
        return prec
    else:
        raise NotImplementedError(f"Preconditioner method '{method}' is not implemented.")
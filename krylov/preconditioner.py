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
    def __init__(self, ilu_op):
        super().__init__()
        self._prec = ilu_op

    @property
    def nnz(self):
        # M = L*U, nnz is sum of non-zeros in factors
        return self._prec.L.nnz + self._prec.U.nnz

    def solve(self, b: torch.Tensor) -> torch.Tensor:
        # CONVERT TO NUMPY for SciPy, then back to torch
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
        self.L_torch = None
        self.U_torch = None

    def _compute_preconditioner(self):
        start_time = time_function()
        with torch.no_grad():
            self.L_torch, self.U_torch, _ = self._model(self._data)
        self.time = time_function() - start_time
        self._computed = True
    
    @property
    def nnz(self):
        if not self._computed: self._compute_preconditioner()
        return self.L_torch.coalesce()._nnz() + self.U_torch.coalesce()._nnz()

    def solve(self, b: torch.Tensor) -> torch.Tensor:
        if not self._computed: self._compute_preconditioner()
        y = torch.triangular_solve(b.unsqueeze(1), self.L_torch, upper=False).values
        x = torch.triangular_solve(y, self.U_torch, upper=True).values
        return x.squeeze(1)

def get_preconditioner(data, method: str, model=None) -> Preconditioner:
    """Factory function to create the specified preconditioner."""
    A_torch = torch.sparse_coo_tensor(
        data.edge_index, data.edge_attr.squeeze(),
        size=(data.num_nodes, data.num_nodes),
        device=data.x.device,
        dtype=torch.float64
    ).coalesce()
    
    if method == "baseline":
        return Preconditioner()
    elif method == "jacobi":
        return Jacobi(A_torch)
    elif method == "ilu":
        # Using SciPy's ILU which is stable in Colab
        A_scipy_csc = torch_sparse_to_scipy(A_torch.cpu()).tocsc()
        start_time = time_function()
        try:
            # Create a linear operator that represents the action of M_inv
            ilu_op = scipy.sparse.linalg.spilu(A_scipy_csc, drop_tol=1e-4, fill_factor=10)
            M = scipy.sparse.linalg.LinearOperator(A_scipy_csc.shape, ilu_op.solve)
            prec = ScipyILU(M)
            # Store factors on the object for nnz calculation
            prec._prec.L = ilu_op.L 
            prec._prec.U = ilu_op.U
        except Exception as e:
            print(f"\nWARNING: SciPy ILU factorization failed for a sample: {e}")
            prec = Preconditioner()
            prec.breakdown = True
        prec.time = time_function() - start_time
        return prec
    elif method == "learned":
        if model is None: raise ValueError("A model must be provided for the 'learned' method.")
        return Learned(model, data)
    else:
        raise NotImplementedError(f"Preconditioner method '{method}' is not implemented.")
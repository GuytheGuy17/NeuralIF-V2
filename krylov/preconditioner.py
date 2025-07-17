# FILE: krylov/preconditioner.py

import torch
import ilupp
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
        # --- THIS IS THE FIX ---
        # Convert to dense before getting the diagonal, which is a supported operation.
        self.inv_diag = 1.0 / A_torch.to_dense().diagonal()

    @property
    def nnz(self):
        return len(self.inv_diag)
    
    def solve(self, b: torch.Tensor) -> torch.Tensor:
        return self.inv_diag * b

class Ilupp(Preconditioner):
    """Wrapper for the ilupp preconditioner."""
    def __init__(self, ilu_prec_obj):
        super().__init__()
        self._prec = ilu_prec_obj

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

def get_preconditioner(data, method: str, model=None, ilupp_kwargs=None) -> Preconditioner:
    """Factory function to create the specified preconditioner."""
    A_torch = torch.sparse_coo_tensor(
        data.edge_index, data.edge_attr.squeeze(),
        size=(data.num_nodes, data.num_nodes),
        device=data.x.device, # Ensure tensor is on the correct device
        dtype=torch.float64
    ).coalesce()
    
    if method == "baseline":
        return Preconditioner()
    elif method == "jacobi":
        return Jacobi(A_torch)
    elif method == "ilu":
        A_scipy_csc = torch_sparse_to_scipy(A_torch.cpu()).tocsc()
        if ilupp_kwargs is None:
            ilupp_kwargs = {'type': 'crout', 'drop_tol': 1e-4, 'fill_factor': 10}
        start_time = time_function()
        try:
            prec_obj = ilupp.ilu(A_scipy_csc, **ilupp_kwargs)
            prec = Ilupp(prec_obj)
        except Exception as e:
            print(f"\nWARNING: ilupp factorization failed for a sample: {e}")
            prec = Preconditioner()
            prec.breakdown = True
        prec.time = time_function() - start_time
        return prec
    elif method == "learned":
        if model is None: raise ValueError("A model must be provided for the 'learned' method.")
        return Learned(model, data)
    else:
        raise NotImplementedError(f"Preconditioner method '{method}' is not implemented.")
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

    def solve(self, b):
        return b # Default is identity (no preconditioning)

class Jacobi(Preconditioner):
    """Jacobi (Diagonal) Preconditioner."""
    def __init__(self, A_torch):
        super().__init__()
        # Extract diagonal and compute its inverse. Ensure it's on CPU for numpy conversion.
        self.inv_diag = 1.0 / A_torch.diagonal().cpu().numpy()

    @property
    def nnz(self):
        return len(self.inv_diag)
    
    def solve(self, b):
        # M_inv * b = D_inv * b
        return self.inv_diag * b

class Ilupp(Preconditioner):
    """Wrapper for the ilupp preconditioner."""
    def __init__(self, ilu_prec_obj):
        super().__init__()
        self._prec = ilu_prec_obj

    @property
    def nnz(self):
        return self._prec.L.nnz + self._prec.U.nnz

    def solve(self, b):
        return self._prec.solve(b)

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
        start_time = time_function()
        with torch.no_grad():
            L_torch, U_torch, _ = self._model(self._data)
        
        self.L_scipy = torch_sparse_to_scipy(L_torch.cpu()).tocsc()
        self.U_scipy = torch_sparse_to_scipy(U_torch.cpu()).tocsc()
        self.time = time_function() - start_time
        self._computed = True
    
    @property
    def nnz(self):
        if not self._computed:
            self._compute_preconditioner()
        return self.L_scipy.nnz + self.U_scipy.nnz

    def solve(self, b):
        if not self._computed:
            self._compute_preconditioner()
        
        y = scipy.sparse.linalg.spsolve_triangular(self.L_scipy, b, lower=True)
        x = scipy.sparse.linalg.spsolve_triangular(self.U_scipy, y, lower=False)
        return x

def get_preconditioner(A_scipy, A_torch, data, method: str, model=None, ilupp_kwargs=None) -> Preconditioner:
    """Factory function to create the specified preconditioner."""
    if method == "baseline":
        return Preconditioner()
    elif method == "jacobi":
        return Jacobi(A_torch)
    elif method == "ilu":
        if ilupp_kwargs is None:
            ilupp_kwargs = {'type': 'crout', 'drop_tol': 1e-4, 'fill_factor': 10}
        start_time = time_function()
        try:
            prec_obj = ilupp.ilu(A_scipy, **ilupp_kwargs)
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
import numpy as np

def arnoldi_iteration(A, num_steps: int):
    """
    Performs Arnoldi iteration to produce an orthonormal basis V and a
    Hessenberg matrix H such that A*V_k = V_{k+1}*H_k.

    Args:
        A: The matrix (can be a SciPy sparse matrix or any object with a .dot method).
        num_steps: The number of iterations to run (the size of the Krylov subspace).

    Returns:
        V: A list of the orthonormal basis vectors.
        H: The (num_steps x num_steps) upper Hessenberg matrix.
    """
    n = A.shape[0]
    
    # Start with a random normalized vector
    b = np.random.rand(n)
    q = b / np.linalg.norm(b)
    
    V = [q]
    H = np.zeros((num_steps + 1, num_steps), dtype=float)

    for k in range(num_steps):
        w = A.dot(V[k]) # Matrix-vector product
        
        # --- Modified Gram-Schmidt ---
        for j in range(k + 1):
            H[j, k] = np.dot(V[j].conj(), w)
            w = w - H[j, k] * V[j]
        
        H[k + 1, k] = np.linalg.norm(w)
        
        # Check for breakdown
        if H[k + 1, k] < 1e-10:
            break
            
        V.append(w / H[k + 1, k])
        
    # Return the basis vectors and the Hessenberg matrix (without the last row)
    return V, H[:num_steps, :num_steps]
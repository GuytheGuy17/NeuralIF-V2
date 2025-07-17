# FILE: apps/synthetic.py

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import torch
import numpy as np
import scipy
from scipy.sparse import coo_matrix, lil_matrix
from apps.data import matrix_to_graph

def generate_fem_like_matrix(grid_dims=(32, 32), varying_conductivity=False):
    """
    Generates a 2D FEM-like sparse matrix.
    """
    height, width = grid_dims
    num_nodes = height * width
    matrix = lil_matrix((num_nodes, num_nodes))

    if varying_conductivity:
        conductivity = 10**(-1 + 2 * np.random.rand(height, width))
    else:
        conductivity = np.ones((height, width))

    def to_1d(i, j):
        return i * width + j

    for i in range(height):
        for j in range(width):
            k = to_1d(i, j)
            diag_val = 0
            if i > 0:
                avg_cond = 0.5 * (conductivity[i, j] + conductivity[i - 1, j]); diag_val += avg_cond; matrix[k, to_1d(i - 1, j)] = -avg_cond
            if i < height - 1:
                avg_cond = 0.5 * (conductivity[i, j] + conductivity[i + 1, j]); diag_val += avg_cond; matrix[k, to_1d(i + 1, j)] = -avg_cond
            if j > 0:
                avg_cond = 0.5 * (conductivity[i, j] + conductivity[i, j - 1]); diag_val += avg_cond; matrix[k, to_1d(i, j - 1)] = -avg_cond
            if j < width - 1:
                avg_cond = 0.5 * (conductivity[i, j] + conductivity[i, j + 1]); diag_val += avg_cond; matrix[k, to_1d(i, j + 1)] = -avg_cond
            
            matrix[k, k] = diag_val
            
    identity_shift = 1e-6 * scipy.sparse.identity(num_nodes)
    matrix = matrix + identity_shift
    return matrix.tocoo()

def generate_sparse_random(n, alpha=1e-3, random_state=0):
    """
    Generates a random sparse SPD matrix, matching the paper's description.
    The final matrix is computed as M = A*A^T + alpha*I. [cite: 655]
    """
    rng = np.random.RandomState(random_state)
    
    # --- CHANGE 1: Sparsity ---
    # The paper specifies problems with ~99% sparsity, meaning 1% non-zero elements. 
    sparsity = 0.01 
    
    nnz = int(sparsity * n ** 2)
    
    # Ensure unique coordinates to avoid duplicate entries
    rows_cols = set()
    while len(rows_cols) < nnz:
        rows_cols.add((rng.randint(0, n), rng.randint(0, n)))
    
    rows, cols = zip(*rows_cols)
    vals = rng.normal(0, 1, size=len(cols)) # Sample from N(0,1) [cite: 653]
    
    M = coo_matrix((vals, (rows, cols)), shape=(n, n))
    I = scipy.sparse.identity(n)
    A = (M @ M.T) + alpha * I
    return A

def main(args):
    """Main function to generate and save the dataset."""
    print(f"Preparing to generate {args.num_samples} samples for dataset '{args.type}'...")
    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(args.num_samples):
        random_state = args.seed + i
        
        if args.type == 'fem':
            matrix = generate_fem_like_matrix(
                grid_dims=(args.grid_size, args.grid_size),
                varying_conductivity=args.varying_conductivity
            )
        else: # 'random' (synthetic)
            # --- CHANGE 2: Alpha value ---
            # Pass the correct alpha from args to the generator.
            matrix = generate_sparse_random(n=args.matrix_size, alpha=args.alpha, random_state=random_state)
            
        num_nodes = matrix.shape[0]
        rng = np.random.RandomState(random_state)
        b = rng.uniform(0, 1, size=num_nodes) # Right-hand side is sampled uniformly [cite: 658]

        graph_data = matrix_to_graph(matrix, b)
        
        save_path = os.path.join(args.output_dir, f'graph_size{args.matrix_size}_sample{i}.pt')
        
        torch.save(graph_data, save_path)
        
        if (i + 1) % 100 == 0 or (i + 1) == args.num_samples:
            print(f"  ... generated and saved sample {i + 1} / {args.num_samples}")

    print("-" * 50)
    print(f"Successfully generated {args.num_samples} graphs in:")
    print(f"{os.path.abspath(args.output_dir)}")
    print("-" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate synthetic graph datasets for GNN preconditioners.")
    
    parser.add_argument('--type', type=str, required=True, choices=['fem', 'random'], help="Type of matrices to generate ('fem' or 'random').")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the generated graph files.")
    parser.add_argument('--num_samples', type=int, default=1000, help="Number of graph samples to generate.")
    parser.add_argument('--seed', type=int, default=42, help="Base random seed for reproducibility.")

    # Arguments for 'fem' type
    parser.add_argument('--grid_size', type=int, default=32, help="Side length of the square grid for FEM (e.g., 32 -> 32x32 grid).")
    parser.add_argument('--varying-conductivity', action='store_true', help="Use varying material properties in FEM generation.")
    
    # Arguments for 'random' (synthetic) type
    parser.add_argument('--matrix_size', type=int, default=1024, help="Matrix size (N) for the 'random' type (NxN).")
    # --- CHANGE 3: Added alpha parameter ---
    # The paper uses an alpha of ~10^-3 
    parser.add_argument('--alpha', type=float, default=1e-3, help="Regularization parameter for synthetic matrix generation.")

    args = parser.parse_args()
    main(args)
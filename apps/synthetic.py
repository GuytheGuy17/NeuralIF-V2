import os
import argparse
import torch
import numpy as np
import scipy
from scipy.sparse import coo_matrix, lil_matrix

# ====================================================================
# FINAL STEP: Import the conversion function from your data.py file
# ====================================================================
from data import matrix_to_graph


# ====================================================================
#  GENERATOR 1: ENHANCED FEM-LIKE MATRICES (Recommended)
# ====================================================================
def generate_fem_like_matrix(grid_dims=(32, 32), varying_conductivity=False):
    """
    Generates a 2D FEM-like sparse matrix.
    ENHANCEMENT: Added 'varying_conductivity' for more realistic, diverse data.
    """
    height, width = grid_dims
    num_nodes = height * width
    matrix = lil_matrix((num_nodes, num_nodes))

    # Optional: create a random conductivity field for realism
    if varying_conductivity:
        conductivity = 10**(-1 + 2 * np.random.rand(height, width)) # Values from 0.1 to 10.0
    else:
        conductivity = np.ones((height, width))

    def to_1d(i, j):
        return i * width + j

    for i in range(height):
        for j in range(width):
            k = to_1d(i, j)
            # Diagonal term accumulates contributions from all neighbors
            diag_val = 0
            # Off-diagonal terms
            if i > 0:
                avg_cond = 0.5 * (conductivity[i, j] + conductivity[i - 1, j])
                diag_val += avg_cond
                matrix[k, to_1d(i - 1, j)] = -avg_cond
            if i < height - 1:
                avg_cond = 0.5 * (conductivity[i, j] + conductivity[i + 1, j])
                diag_val += avg_cond
                matrix[k, to_1d(i + 1, j)] = -avg_cond
            if j > 0:
                avg_cond = 0.5 * (conductivity[i, j] + conductivity[i, j - 1])
                diag_val += avg_cond
                matrix[k, to_1d(i, j - 1)] = -avg_cond
            if j < width - 1:
                avg_cond = 0.5 * (conductivity[i, j] + conductivity[i, j + 1])
                diag_val += avg_cond
                matrix[k, to_1d(i, j + 1)] = -avg_cond
            
            matrix[k, k] = diag_val
            
    # Make the matrix SPD by adding a small diagonal shift
    identity_shift = 1e-6 * scipy.sparse.identity(num_nodes)
    matrix = matrix + identity_shift
    return matrix.tocoo()

# ====================================================================
#  GENERATOR 2: ORIGINAL RANDOM SPD MATRICES
# ====================================================================
def generate_sparse_random(n, alpha=1e-4, random_state=0):
    """Generates a random sparse SPD matrix (your original method)."""
    rng = np.random.RandomState(random_state)
    sparsity = 10e-4
    nnz = int(sparsity * n ** 2)
    rows, cols = zip(*set(zip(rng.randint(0, n, size=nnz), rng.randint(0, n, size=nnz))))
    vals = rng.normal(0, 1, size=len(cols))
    M = coo_matrix((vals, (rows, cols)), shape=(n, n))
    I = scipy.sparse.identity(n)
    A = (M @ M.T) + alpha * I
    return A

# ====================================================================
#  MAIN SCRIPT LOGIC
# ====================================================================
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
        else: # 'random'
            matrix = generate_sparse_random(n=args.matrix_size, random_state=random_state)
            
        num_nodes = matrix.shape[0]
        rng = np.random.RandomState(random_state)
        b = rng.uniform(0, 1, size=num_nodes)

        # Use the imported function to convert the matrix to a PyG Data object
        graph_data = matrix_to_graph(matrix, b)
        
        save_path = os.path.join(args.output_dir, f'graph_{i}.pt')
        torch.save(graph_data, save_path)
        
        if (i + 1) % 100 == 0 or (i + 1) == args.num_samples:
            print(f"  ... generated and saved sample {i + 1} / {args.num_samples}")

    print("-" * 50)
    print(f"Successfully generated {args.num_samples} graphs in:")
    print(f"{os.path.abspath(args.output_dir)}")
    print("-" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate synthetic graph datasets for GNN preconditioners.")
    
    parser.add_argument('--type', type=str, required=True, choices=['fem', 'random'],
                        help="Type of matrices to generate ('fem' or 'random').")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save the generated graph files.")
    parser.add_argument('--num_samples', type=int, default=1000,
                        help="Number of graph samples to generate.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Base random seed for reproducibility.")

    # Arguments for 'fem' type
    parser.add_argument('--grid_size', type=int, default=32,
                        help="Side length of the square grid for FEM (e.g., 32 -> 32x32 grid).")
    parser.add_argument('--varying-conductivity', action='store_true',
                        help="Use varying material properties in FEM generation for more realistic data.")
    
    # Arguments for 'random' type
    parser.add_argument('--matrix_size', type=int, default=1024,
                        help="Matrix size (N) for the 'random' type (NxN).")

    args = parser.parse_args()
    
    # A small check to ensure matrix sizes are consistent if needed
    if args.type == 'fem':
        args.matrix_size = args.grid_size * args.grid_size

    main(args)

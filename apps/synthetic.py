import os
import argparse
import torch
import numpy as np
import scipy
from scipy.sparse import coo_matrix, lil_matrix
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix


# ====================================================================
#  GENERATOR 1: NEW FEM-LIKE MATRICES (Recommended)
# ====================================================================

def generate_fem_like_matrix(grid_dims=(32, 32)):
    """Generates a 2D FEM-like sparse matrix by discretizing the Poisson equation."""
    height, width = grid_dims
    num_nodes = height * width
    matrix = lil_matrix((num_nodes, num_nodes))

    def to_1d(i, j):
        return i * width + j

    for i in range(height):
        for j in range(width):
            k = to_1d(i, j)
            matrix[k, k] = 4.0 # From the 5-point stencil diagonal
            if i > 0: matrix[k, to_1d(i - 1, j)] = -1.0
            if i < height - 1: matrix[k, to_1d(i + 1, j)] = -1.0
            if j > 0: matrix[k, to_1d(i, j - 1)] = -1.0
            if j < width - 1: matrix[k, to_1d(i, j + 1)] = -1.0
            
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
    rows = rng.randint(0, n, size=nnz)
    cols = rng.randint(0, n, size=nnz)
    
    uniques = set(zip(rows, cols))
    rows, cols = zip(*uniques)
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
    print(f"Preparing to generate {args.num_samples} samples...")
    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(args.num_samples):
        # Use a unique random seed for each sample
        random_state = args.seed + i
        
        # --- 1. Generate the base matrix ---
        if args.type == 'fem':
            matrix = generate_fem_like_matrix(grid_dims=(args.grid_size, args.grid_size))
        elif args.type == 'random':
            matrix = generate_sparse_random(n=args.matrix_size, random_state=random_state)
        else:
            raise ValueError("Unknown dataset type specified.")
            
        # --- 2. Generate the 'b' vector ---
        num_nodes = matrix.shape[0]
        rng = np.random.RandomState(random_state)
        b = rng.uniform(0, 1, size=num_nodes)

        # --- 3. Convert to PyG graph and save ---
        graph_data = matrix_to_pyg_data(matrix, b)
        
        save_path = os.path.join(args.output_dir, f'graph_{i}.pt')
        torch.save(graph_data, save_path)
        
        if (i + 1) % 50 == 0:
            print(f"  ... generated and saved sample {i + 1} / {args.num_samples}")

    print("-" * 40)
    print(f"Successfully generated {args.num_samples} graphs in:")
    print(f"{os.path.abspath(args.output_dir)}")
    print("-" * 40)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate synthetic graph datasets.")
    
    parser.add_argument('--type', type=str, required=True, choices=['fem', 'random'],
                        help="Type of matrices to generate ('fem' or 'random').")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save the generated graph files.")
    parser.add_argument('--num_samples', type=int, default=1000,
                        help="Number of graph samples to generate.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Base random seed for reproducibility.")

    # Arguments specific to 'fem' type
    parser.add_argument('--grid_size', type=int, default=32,
                        help="Size of one side of the square grid for FEM matrices (e.g., 32 for a 32x32 grid).")
    
    # Arguments specific to 'random' type
    parser.add_argument('--matrix_size', type=int, default=1024,
                        help="Size of the matrix for the 'random' type.")

    args = parser.parse_args()
    main(args)

if __name__ == '__main__':
    # create the folders and subfolders where the data is stored
    os.makedirs(f'./data/Random/train', exist_ok=True)
    os.makedirs(f'./data/Random/val', exist_ok=True)
    os.makedirs(f'./data/Random/test', exist_ok=True)
    
    # create 10k dataset
    n = 10_000
    alpha=10e-4
    
    create_dataset(n, 1000, alpha=alpha, mode='train', rs=0, graph=True, solution=True)
    create_dataset(n, 10, alpha=alpha, mode='val', rs=10000, graph=True)
    create_dataset(n, 100, alpha=alpha, mode='test', rs=103600, graph=True)

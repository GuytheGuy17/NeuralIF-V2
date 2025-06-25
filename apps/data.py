import os
import glob
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix

# ====================================================================
#  1. EFFICIENT MATRIX-TO-GRAPH CONVERSION
#     (Replaces your original 'matrix_to_graph_sparse')
# ====================================================================
def matrix_to_graph(scipy_matrix, b_vector):
    """
    Efficiently converts a SciPy sparse matrix and a vector 'b' into a 
    PyTorch Geometric Data object.
    """
    # Use the optimized PyG utility for conversion. It's much faster.
    edge_index, edge_attr = from_scipy_sparse_matrix(scipy_matrix)

    # Use the 'b' vector as the node features, ensuring it has a feature dimension.
    node_features = torch.tensor(b_vector, dtype=torch.float).view(-1, 1)

    # Assemble the Data object
    data = Data(x=node_features, edge_index=edge_index.long(), edge_attr=edge_attr.float())
    return data

# ====================================================================
#  2. SIMPLIFIED AND GENERALIZED FOLDERDATASET
#     (Replaces your original FolderDataset)
# ====================================================================
class FolderDataset(torch.utils.data.Dataset):
    """
    A generic dataset that loads all '.pt' graph files from a specified folder.
    The complex filename filtering has been removed to make it more general.
    """
    def __init__(self, folder_path):
        super().__init__()
        
        # Use glob to find all files ending in .pt in the given folder
        self.files = sorted(glob.glob(os.path.join(folder_path, '*.pt')))
        
        if not self.files:
            raise FileNotFoundError(f"CRITICAL: No '.pt' files were found in the directory: {folder_path}\n"
                                    f"Please run the 'generate_data.py' script first to create the dataset.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load a single graph file.
        # weights_only=False is used for compatibility with newer PyTorch versions.
        return torch.load(self.files[idx], weights_only=False)

# ====================================================================
#  3. FLEXIBLE DATALOADER (The Main Change)
#     (Replaces your original get_dataloader)
# ====================================================================
def get_dataloader(dataset_name, batch_size=1, mode="train"):
    """
    Creates a DataLoader for a given dataset name (e.g., "FEM", "Random")
    by loading graph files from the corresponding directory.
    """
    # Construct the path automatically based on the dataset name and mode
    folder_path = os.path.join('./data', dataset_name, mode)
    
    # Instantiate our generalized FolderDataset
    dataset = FolderDataset(folder_path=folder_path)

    # Set shuffle=True for training, False otherwise
    should_shuffle = (mode == "train")

    print(f"Successfully created a '{mode}' dataloader for the '{dataset_name}' dataset.")
    print(f" -> Loading {len(dataset)} samples from: {os.path.abspath(folder_path)}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=should_shuffle)


# This function is not strictly needed anymore if your generation script
# uses the new `matrix_to_graph`, but we can keep it for other utilities.
def graph_to_matrix(data, normalize=False):
    """Converts a PyG Data object back to a sparse matrix and vector."""
    A = torch.sparse_coo_tensor(data.edge_index, data.edge_attr.squeeze(), requires_grad=False)
    b = data.x.squeeze()
    
    if normalize:
        b = b / torch.linalg.norm(b)
        
    return A, b
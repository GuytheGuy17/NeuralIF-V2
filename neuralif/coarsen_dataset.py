import torch
import torch_geometric
from torch_geometric.nn.pool import graclus, avg_pool
import glob
import os

def coarsen_graph(data):
    """
    Applies one level of Graclus clustering to coarsen a graph.
    """
    # 1. Get the cluster assignments for each node
    clusters = graclus(data.edge_index, num_nodes=data.num_nodes)
    
    # 2. Pool the graph data based on the clusters
    # The avg_pool function automatically coarsens the graph structure
    # and averages the features of nodes within each cluster.
    coarsened_data = avg_pool(clusters, data)
    
    return coarsened_data


def main():
    # --- Configuration ---
    # The folder containing your original, large graphs
    input_dataset_dir = "/content/drive/MyDrive/My_Thesis_Datasets/fem_mixed"
    # The new folder where the smaller, coarsened graphs will be saved
    output_dataset_dir = "/content/drive/MyDrive/My_Thesis_Datasets/fem_mixed_coarsened"

    print(f"Starting coarsening process...")
    print(f"Input Directory: {input_dataset_dir}")
    print(f"Output Directory: {output_dataset_dir}")

    # Process both 'train' and 'val' subdirectories
    for mode in ['train', 'val']:
        input_path = os.path.join(input_dataset_dir, mode)
        output_path = os.path.join(output_dataset_dir, mode)
        os.makedirs(output_path, exist_ok=True)
        
        files = glob.glob(os.path.join(input_path, '*.pt'))
        print(f"\nFound {len(files)} graphs in '{mode}' set. Coarsening...")

        for i, file_path in enumerate(files):
            original_data = torch.load(file_path)
            
            # Coarsen the graph
            coarsened_data = coarsen_graph(original_data)
            
            # Save the new, smaller graph
            base_filename = os.path.basename(file_path)
            new_save_path = os.path.join(output_path, base_filename)
            torch.save(coarsened_data, new_save_path)

            if (i + 1) % 100 == 0:
                print(f"  ...processed {i+1}/{len(files)} graphs.")
    
    print("\n--- ✅ Coarsening Complete ---")


if __name__ == '__main__':
    main()
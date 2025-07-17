import argparse
import os
import datetime
import numpy as np
import torch
import json
from tqdm import tqdm

from krylov.cg import conjugate_gradient, preconditioned_conjugate_gradient
from krylov.gmres import gmres
from krylov.arnoldi import arnoldi_iteration
from krylov.preconditioner import get_preconditioner
from neuralif.models import NeuralIF, NeuralPCG, PreCondNet, LearnedLU
from neuralif.utils import torch_sparse_to_scipy, time_function
from neuralif.logger import TestResults
from apps.data import matrix_to_graph, get_dataloader


def load_checkpoint(model_class, checkpoint_dir, device):
    """Loads a model's configuration and weights from a checkpoint directory."""
    config_path = os.path.join(checkpoint_dir, "config.json")
    weights_path = os.path.join(checkpoint_dir, "best_model.pt") # Assuming you want the best model

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights file not found at: {weights_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Instantiate the model with all the correct parameters from the config file
    model = model_class(**config)
    
    # Load the saved weights
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Successfully loaded model from {checkpoint_dir}")
    return model

def run_solver(solver_name, A, b, M, settings, x_true=None):
    """Helper function to dispatch to the correct solver. Operates on torch tensors."""
    if solver_name == "cg":
        if M.__class__.__name__ == 'Preconditioner':
             return conjugate_gradient(A, b, x_true=x_true, **settings)
        else:
             return preconditioned_conjugate_gradient(A, b, M=M, x_true=x_true, **settings)
    else:
        raise NotImplementedError(f"Solver '{solver_name}' not implemented.")

def test(config):
    """Main testing function."""
    device = torch.device(f"cuda:{config['device']}" if torch.cuda.is_available() and config['device'] is not None else "cpu")
    print(f"\nUsing device: {device}")

    model = None
    methods_to_test = ["baseline", "jacobi", "ilu"]
    
    if config['model'] != "none":
        methods_to_test = ["learned"]
        model_map = {"neuralif": NeuralIF, "neuralpcg": NeuralPCG, "precondnet": PreCondNet, "learnedlu": LearnedLU}
        model_class = model_map.get(config['model'])
        if model_class is None: raise NotImplementedError(f"Model {config['model']} not available.")
        
        # --- THIS IS THE FIX ---
        # Call the new function to properly load the model and its configuration
        model = load_checkpoint(model_class, config['checkpoint'], device)
    
    test_loader = get_dataloader(config['dataset'], batch_size=1, mode=config['subset'])
    
    results_loggers = {
        method: TestResults(method=method, dataset=config['dataset'], folder=config['folder'], solver=config['solver'])
        for method in methods_to_test
    }

    print(f"\nTesting on {len(test_loader.dataset)} samples...")
    print(f"-> Methods: {methods_to_test}")

    for data in tqdm(test_loader, desc="Testing Progress"):
        data = data.to(device)
        A_torch = torch.sparse_coo_tensor(
            data.edge_index, data.edge_attr.squeeze(),
            size=(data.num_nodes, data.num_nodes),
            dtype=torch.float64, device=device
        ).coalesce()
        
        b = data.x[:, 0].squeeze().to(torch.float64)
        b_norm = torch.linalg.norm(b)
        b /= b_norm
        x_true = data.s.to(torch.float64).squeeze() / b_norm if hasattr(data, 's') else None

        for method in methods_to_test:
            prec = get_preconditioner(data, method, model=model)
            if prec.breakdown: continue

            solver_settings = {"max_iter": 10_000, "rtol": 1e-6}
            
            solver_start_time = time_function()
            res, err = run_solver(config['solver'], A_torch, b, M=prec, settings=solver_settings, x_true=x_true)
            solver_time = time_function() - solver_start_time
            
            logger = results_loggers[method]
            logger.log_solve(
                n=A_torch.shape[0], solver_time=solver_time, p_time=prec.time,
                solver_iterations=len(res) - 1, solver_error=[e.item() for e in err],
                solver_residual=[r.item() for r in res], overhead=0
            )
            logger.log(nnz_a=A_torch._nnz(), nnz_p=prec.nnz)

    print("\n--- TEST RESULTS ---")
    for method, logger in results_loggers.items():
        print(f"\n--- Summary for: {method} ---")
        logger.print_summary()
        if config['save']:
            logger.save_results()

def main():
    parser = argparse.ArgumentParser(description="Testing script for NeuralIF.")
    parser.add_argument("--device", type=int, help="GPU device ID.")
    parser.add_argument("--model", type=str, default="none", help="Model to test ('none' for baselines).")
    parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint directory for a trained model.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--subset", type=str, default="test", help="Dataset subset to use.")
    parser.add_argument("--solver", type=str, default="cg", choices=["cg"], help="Krylov solver to use.")
    parser.add_argument("--name", type=str, default=f"test_run_{datetime.datetime.now():%Y-%m-%d_%H-%M}", help="Name for the results folder.")
    parser.add_argument("--save", action='store_true', help="Save results and plots.")
    args = parser.parse_args()

    # Check for required arguments
    if args.model != "none" and args.checkpoint is None:
        parser.error("--checkpoint is required when specifying a model.")

    config = vars(args)
    config['folder'] = os.path.join("results", config['name'])
    if config['save']:
        os.makedirs(config['folder'], exist_ok=True)
        with open(os.path.join(config['folder'], "test_config.json"), 'w') as f:
            json.dump(config, f, indent=4)
    test(config)

if __name__ == "__main__":
    main()
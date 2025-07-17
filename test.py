import argparse
import os
import datetime
import numpy as np
import scipy.sparse
import torch
import json
from tqdm import tqdm # For a helpful progress bar

from krylov.cg import conjugate_gradient, preconditioned_conjugate_gradient
from krylov.gmres import gmres
from krylov.arnoldi import arnoldi_iteration # Import the new function
from krylov.preconditioner import get_preconditioner
from neuralif.models import NeuralIF, NeuralPCG, PreCondNet, LearnedLU
from neuralif.utils import torch_sparse_to_scipy, time_function
from neuralif.logger import TestResults
from apps.data import matrix_to_graph, get_dataloader

def run_solver(solver_name, A, b, M, settings, x_true=None):
    """Helper function to dispatch to the correct solver."""
    if solver_name == "cg":
        if M is None or M.__class__.__name__ == 'Preconditioner': # Unpreconditioned case
             return conjugate_gradient(A, b, x_true=x_true, **settings)
        else:
             return preconditioned_conjugate_gradient(A, b, M=M, x_true=x_true, **settings)
    elif solver_name == "gmres":
        return gmres(A, b, M=M, x_true=x_true, **settings)
    else:
        raise NotImplementedError(f"Solver '{solver_name}' not implemented.")

def test(config):
    """
    Main testing function, rewritten for performance and clarity.
    """
    device = torch.device(f"cuda:{config['device']}" if torch.cuda.is_available() and config['device'] is not None else "cpu")
    print(f"\nUsing device: {device}")

    # --- Load Model (if applicable) ---
    model = None
    if config['model'] != "none":
        model_map = {"neuralif": NeuralIF, "neuralpcg": NeuralPCG, "precondnet": PreCondNet, "learnedlu": LearnedLU}
        model_class = model_map.get(config['model'])
        if model_class is None: raise NotImplementedError(f"Model {config['model']} not available.")
        
        print(f"Loading model: {model_class.__name__}")
        # This logic should be adapted if you have a real checkpoint to load
        model = model_class(**{"latent_size": 16, "message_passing_steps": 4})
        model.to(device).eval()
        print("Model loaded and in eval mode.")

    # --- Setup Data and Methods ---
    test_loader = get_dataloader(config['dataset'], batch_size=1, mode=config['subset'])
    
    # Determine which methods to run
    methods_to_test = []
    if not config['analyze_arnoldi_only']:
        methods_to_test = ["baseline", "jacobi", "ilu"] if model is None else ["learned"]
    
    # --- Initialize Result Loggers ---
    results_loggers = {
        method: TestResults(method=method, dataset=config['dataset'], folder=config['folder'], solver=config['solver'])
        for method in methods_to_test
    }
    if config['analyze_arnoldi']:
        results_loggers['arnoldi_eigs'] = TestResults(method='arnoldi_eigs', dataset=config['dataset'], folder=config['folder'])

    print(f"\nTesting on {len(test_loader.dataset)} samples...")
    if methods_to_test: print(f"-> Preconditioner methods: {methods_to_test}")
    if config['analyze_arnoldi']: print("-> Analysis: Arnoldi Iteration for Eigenvalues")

    # --- Main Test Loop ---
    for data in tqdm(test_loader, desc="Testing Progress"):
        data = data.to(device)
        
        # Convert to CPU formats once per sample for baselines
        A_torch_cpu = torch.sparse_coo_tensor(
            data.edge_index, data.edge_attr.squeeze(),
            size=(data.num_nodes, data.num_nodes),
            dtype=torch.float64, requires_grad=False
        ).cpu().coalesce()
        
        A_scipy_cpu_csc = torch_sparse_to_scipy(A_torch_cpu).tocsc()
        
        # --- Run Arnoldi Analysis (if enabled) ---
        if config['analyze_arnoldi']:
            _, H = arnoldi_iteration(A_scipy_cpu_csc, num_steps=config['arnoldi_steps'])
            eigenvalues = np.linalg.eigvals(H)
            results_loggers['arnoldi_eigs'].log_eigenval_dist(torch.from_numpy(np.real(eigenvalues)))

        # --- Run Preconditioner Tests ---
        if not methods_to_test:
            continue

        b_cpu = data.x[:, 0].squeeze().cpu().to(torch.float64)
        b_norm = torch.linalg.norm(b_cpu)
        b_cpu /= b_norm
        x_true_cpu = data.s.cpu().to(torch.float64).squeeze() / b_norm if hasattr(data, 's') else None
        
        solver_settings = {"max_iter": 10_000, "rtol": 1e-6}

        for method in methods_to_test:
            prec = get_preconditioner(A_scipy_cpu_csc, A_torch_cpu, data, method, model=model)
            if prec.breakdown: continue

            res, err = run_solver(
                config['solver'], A_scipy_cpu_csc, b_cpu.numpy(),
                M=prec, settings=solver_settings,
                x_true=x_true_cpu.numpy() if x_true_cpu is not None else None
            )

            logger = results_loggers[method]
            logger.log_solve(n=A_torch_cpu.shape[0], solver_time=time_function(), p_time=prec.time,
                             solver_iterations=len(res) - 1, solver_error=err, solver_residual=res, overhead=0)
            logger.log(nnz_a=A_torch_cpu._nnz(), nnz_p=prec.nnz)

    # --- Print and Save Summaries ---
    print("\n--- TEST RESULTS ---")
    for method, logger in results_loggers.items():
        print(f"\n--- Summary for: {method} ---")
        if 'eigs' in method:
            # Plot the final eigenvalue distribution from the Arnoldi analysis
            logger.plot_eigvals(torch.cat(logger.distribution), name="final_distribution")
            print(f"Eigenvalue distribution plot saved to {config['folder']}")
        else:
            logger.print_summary()
        if config['save']:
            logger.save_results()

def main():
    parser = argparse.ArgumentParser(description="Optimised testing script for NeuralIF.")
    parser.add_argument("--device", type=int, help="GPU device ID.")
    parser.add_argument("--model", type=str, default="none", help="Model to test ('none' for baselines).")
    parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint directory.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--subset", type=str, default="test", help="Dataset subset to use ('train', 'val', or 'test').")
    parser.add_argument("--solver", type=str, default="cg", choices=["cg", "gmres"], help="Krylov solver to use.")
    parser.add_argument("--name", type=str, default=f"test_run_{datetime.datetime.now():%Y-%m-%d_%H-%M}", help="Name for the results folder.")
    parser.add_argument("--save", action='store_true', help="Save results and plots.")
    
    # New arguments for Arnoldi analysis
    parser.add_argument("--analyze-arnoldi", action='store_true', help="Run Arnoldi iteration to analyze eigenvalues.")
    parser.add_argument("--arnoldi-steps", type=int, default=50, help="Number of steps for Arnoldi iteration.")
    parser.add_argument("--analyze-arnoldi-only", action='store_true', help="Run only the Arnoldi analysis, skipping other tests.")

    args = parser.parse_args()
    config = vars(args)
    config['folder'] = os.path.join("results", config['name'])
    if config['save']:
        os.makedirs(config['folder'], exist_ok=True)
        with open(os.path.join(config['folder'], "test_config.json"), 'w') as f:
            json.dump(config, f, indent=4)

    test(config)

if __name__ == "__main__":
    main()
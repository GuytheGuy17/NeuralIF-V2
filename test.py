import argparse
import os
import datetime
import numpy as np
import scipy
import scipy.sparse
import torch
import json

from krylov.cg import conjugate_gradient, preconditioned_conjugate_gradient
from krylov.gmres import gmres
from krylov.preconditioner import get_preconditioner

from neuralif.models import NeuralIF, NeuralPCG, PreCondNet, LearnedLU
from neuralif.utils import torch_sparse_to_scipy, time_function
from neuralif.logger import TestResults

# CORRECTED: Import 'matrix_to_graph' instead of the old name
from apps.data import matrix_to_graph, get_dataloader


@torch.inference_mode()
def test(model, test_loader, device, folder, save_results=False, dataset="random", solver="cg"):
    # This function remains unchanged
    if save_results:
        os.makedirs(folder, exist_ok=False)

    print()
    print(f"Test:\t{len(test_loader.dataset)} samples")
    print(f"Solver:\t{solver} solver")
    print()
    
    if model is None:
        methods = ["baseline", "jacobi", "ilu"]
    else:
        assert solver in ["cg", "gmres"], "Data-driven method only works with CG or GMRES"
        methods = ["learned"]
    
    if solver == "direct":
        methods = ["direct"]
    
    for method in methods:
        print(f"Testing {method} preconditioner")
        
        test_results = TestResults(method, dataset, folder,
                                   model_name= f"\n{model.__class__.__name__}" if method == "learned" else "",
                                   target=1e-6,
                                   solver=solver)
        
        for sample, data in enumerate(test_loader):
            plot = save_results and sample == (len(test_loader.dataset) - 1)
            
            start = time_function()
            
            data = data.to(device)
            prec = get_preconditioner(data, method, model=model)
            
            p_time = prec.time
            breakdown = prec.breakdown
            nnzL = prec.nnz
            
            stop = time_function()
            
            A = torch.sparse_coo_tensor(data.edge_index, data.edge_attr.squeeze(),
                                        dtype=torch.float64,
                                        requires_grad=False).to("cpu").to_sparse_csr()
            
            b = data.x[:, 0].squeeze().to("cpu").to(torch.float64)
            b_norm = torch.linalg.norm(b)
            
            b = b / b_norm
            solution = data.s.to("cpu").to(torch.float64).squeeze() / b_norm if hasattr(data, "s") else None
            
            overhead = (stop - start) - (p_time)
            
            start_solver = time_function()
            
            solver_settings = {
                "max_iter": 10_000,
                "x0": None
            }
            
            if breakdown:
                res = []
            
            elif solver == "direct":
                A_ = torch.sparse_coo_tensor(data.edge_index, data.edge_attr.squeeze(),
                                              dtype=torch.float64, requires_grad=False)
                A_s = torch_sparse_to_scipy(A_).tocsr()
                start_solver = time_function()
                _ = scipy.sparse.linalg.spsolve(A_s, b.numpy())
                res = [(torch.Tensor([0]), torch.Tensor([0]))] * 2
            
            elif solver == "cg" and method == "baseline":
                res, _ = conjugate_gradient(A, b, x_true=solution,
                                            rtol=test_results.target, **solver_settings)
            
            elif solver == "cg":
                res, _ = preconditioned_conjugate_gradient(A, b, M=prec, x_true=solution,
                                                           rtol=test_results.target, **solver_settings)
                
            elif solver == "gmres":
                res, _ = gmres(A, b, M=prec, x_true=solution,
                               **solver_settings, plot=plot,
                               atol=test_results.target,
                               left=False)
            
            stop_solver = time_function()
            solver_time = (stop_solver - start_solver)
            
            test_results.log_solve(A.shape[0], solver_time, len(res) - 1,
                                   np.array([r[0].item() for r in res]),
                                   np.array([r[1].item() for r in res]),
                                   p_time, overhead)
            
            nnzA = A._nnz()
            test_results.log(nnzA, nnzL, plot=plot)
        
        if save_results:
            test_results.save_results()
        
        test_results.print_summary()


def load_checkpoint(model, args, device):
    # This function remains unchanged
    checkpoint_path = args.checkpoint
    
    if checkpoint_path == "latest":
        # Logic to find the latest checkpoint
        results_dir = "./results/"
        all_dirs = sorted([os.path.join(results_dir, d) for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))])
        
        config = None
        for d in reversed(all_dirs):
            dir_contents = os.listdir(d)
            if "config.json" in dir_contents and "best_model.pt" in dir_contents:
                with open(os.path.join(d, "config.json")) as f:
                    config = json.load(f)
                
                if config.get("model") != args.model:
                    config = None
                    continue
                
                checkpoint_path = os.path.join(d, "best_model.pt")
                break
        if config is None:
            raise FileNotFoundError("Could not find a compatible 'latest' checkpoint.")

    else:
        with open(os.path.join(checkpoint_path, "config.json")) as f:
            config = json.load(f)
        checkpoint_path = os.path.join(checkpoint_path, f"{args.weights}.pt")

    if args.model == "neuralif":
        config["drop_tol"] = args.drop_tol
    
    model = model(**config)
    print(f"Loading checkpoint from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device)))
    
    return model


def warmup(model, device):
    # set testing parameters
    model.to(device)
    model.eval()
    
    # run model warmup
    test_size = 1_000
    matrix = scipy.sparse.coo_matrix((np.ones(test_size), (np.arange(test_size), np.arange(test_size))))
    
    # CORRECTED: Call 'matrix_to_graph'
    data = matrix_to_graph(matrix, torch.ones(test_size))
    data.to(device)
    _ = model(data)
    
    print("Model warmup done...")


def argparser():
    # This function remains unchanged
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--device", type=int, required=False)
    parser.add_argument("--model", type=str, required=False, default="none")
    parser.add_argument("--checkpoint", type=str, required=False)
    parser.add_argument("--weights", type=str, required=False, default="best_model")
    parser.add_argument("--drop_tol", type=float, default=0)
    parser.add_argument("--solver", type=str, default="cg")
    parser.add_argument("--dataset", type=str, required=False, default="random")
    parser.add_argument("--subset", type=str, required=False, default="test")
    parser.add_argument("--n", type=int, required=False, default=0)
    parser.add_argument("--samples", type=int, required=False, default=None)
    parser.add_argument("--save", action='store_true', default=False)
    return parser.parse_args()


def main():
    # This function remains unchanged
    args = argparser()
    
    test_device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device is not None else "cpu")
    
    folder = "results/" + (args.name if args.name else datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    
    print()
    print(f"Using device: {test_device}")
    
    model_map = {
        "nif": NeuralIF, "neuralif": NeuralIF,
        "lu": LearnedLU, "learnedlu": LearnedLU,
        "neural_pcg": NeuralPCG, "neuralpcg": NeuralPCG,
        "precondnet": PreCondNet
    }
    
    if args.model in model_map:
        print(f"Use model: {model_map[args.model].__name__}")
        model = load_checkpoint(model_map[args.model], args, test_device)
        warmup(model, test_device)
    elif args.model == "none":
        print("Running non-data-driven baselines")
        model = None
    else:
        raise NotImplementedError(f"Model {args.model} not available.")
        
    spd = args.solver in ["cg", "direct"]
    testdata_loader = get_dataloader(args.dataset, batch_size=1, mode=args.subset)

    test(model, testdata_loader, test_device, folder,
         save_results=args.save, dataset=args.dataset, solver=args.solver)


if __name__ == "__main__":
    main()
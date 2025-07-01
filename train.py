import os
import datetime
import argparse
import pprint
import torch
import torch_geometric
import time

from apps.data import get_dataloader, graph_to_matrix

from neuralif.utils import count_parameters, save_dict_to_file, condition_number, eigenval_distribution, gershgorin_norm
from neuralif.logger import TrainResults, TestResults
from neuralif.loss import loss
from neuralif.models import NeuralPCG, NeuralIF, PreCondNet, LearnedLU

from krylov.cg import preconditioned_conjugate_gradient
from krylov.gmres import gmres
from krylov.preconditioner import LearnedPreconditioner

@torch.no_grad()
def validate(model, validation_loader, solve=False, solver="cg", **kwargs):
    model.eval()
            
    acc_loss = 0.0
    num_loss = 0
    acc_solver_iters = 0.0
    # Ensure the model is on the correct device
    # This is important for the solver-based validation
    global device 
    
    for i, data in enumerate(validation_loader):
        data = data.to(device)
        
        A, b = graph_to_matrix(data)
        
        if solve:
            
            original_device = next(model.parameters()).device
            try:
                # Move the model to CPU for the solver-based validation
                model.to("cpu")
                
                with torch.inference_mode():
                    preconditioner = LearnedPreconditioner(data.to("cpu"), model)
                
                A = A.to("cpu").to(torch.float64)
                b = b.to("cpu").to(torch.float64)
                
                if solver == "cg":
                    l, x_hat = preconditioned_conjugate_gradient(A, b, M=preconditioner, x0=None, rtol=1e-6, max_iter=1000)
                elif solver == "gmres":
                    l, x_hat = gmres(A, b, M=preconditioner, x0=None, atol=1e-6, max_iter=1000, left=False)
                else:
                    raise NotImplementedError("Solver not implemented choose between CG and GMRES!")
                
                acc_solver_iters += len(l) - 1

            finally:
                # This GUARANTEES the model is moved back to its original device (e.g., the GPU)
                model.to(original_device)
        
        else:
            # This part for loss-based validation remains the same
            output, _, _ = model(data)
            l = loss(data, output, config="frobenius")
            acc_loss += l.item()
            num_loss += 1
            
    if solve:
        print(f"Validation\t iterations:\t{acc_solver_iters / len(validation_loader):.2f}")
        return acc_solver_iters / len(validation_loader)
    else:
        print(f"Validation loss:\t{acc_loss / num_loss:.2f}")
        return acc_loss / len(validation_loader)

def main(config):
    if config["save"]:
        os.makedirs(folder, exist_ok=True)
        save_dict_to_file(config, os.path.join(folder, "config.json"))
    
    # global seed-ish
    torch_geometric.seed_everything(config["seed"])
    
    # args for the model
    model_args = {k: config[k] for k in ["latent_size", "message_passing_steps", "skip_connections",
                                         "augment_nodes", "global_features", "decode_nodes",
                                         "normalize_diag", "activation", "aggregate", "graph_norm",
                                         "two_hop", "edge_features", "normalize"]
                  if k in config}
    
    # run the GMRES algorithm instead of CG (?)
    gmres = False
    
    # Create model
    if config["model"] == "neuralpcg":
        model = NeuralPCG(**model_args)
    
    elif config["model"] == "nif" or config["model"] == "neuralif" or config["model"] == "inf":
        model = NeuralIF(**model_args)
        
    elif config["model"] == "precondnet":
        model = PreCondNet(**model_args)
        
    elif config["model"] == "lu" or config["model"] == "learnedlu":
        gmres = True
        model = LearnedLU(**model_args)
        
    else:
        raise NotImplementedError
    
    if config["load_model_path"]:
        print(f"Loading model weights from: {config['load_model_path']}")
        model.load_state_dict(torch.load(config['load_model_path'], map_location=device))

    
    model.to(device)
    
    print(f"Number params in model: {count_parameters(model)}")
    print()
    
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20)
    
    train_loader = get_dataloader(dataset_name=config["dataset"], 
                                  batch_size=config["batch_size"], 
                                  mode="train")
    # If a specific validation path is given, use it. Otherwise, use the main dataset path.
    validation_path = config["validation_dataset"] if config["validation_dataset"] is not None else config["dataset"]

    validation_loader = get_dataloader(dataset_name=validation_path,
                                        batch_size=1,
                                        mode="val")
    logger = TrainResults(folder)
    best_val = float('inf')
    # todo: compile the model
    # compiled_model = torch.compile(model, mode="reduce-overhead")
    # model = torch_geometric.compile(model, mode="reduce-overhead")
    
    total_it = 0
    
    # Train loop
    for epoch in range(config["num_epochs"]):
        running_loss = 0.0
        grad_norm = 0.0
        
        start_epoch = time.perf_counter()
        
        for it, data in enumerate(train_loader):
                        # Print statistics for the current graph
            num_nodes = data.num_nodes
            num_edges = data.num_edges
            print(f"--- Epoch {epoch+1}, Iteration {it+1}/{len(train_loader)} ---")
            print(f"Graph size: {num_nodes} nodes, {num_edges} edges.")

            # Print current GPU memory usage
            if device.type == 'cuda':
                allocated_mem = torch.cuda.memory_allocated(device) / (1024**2) # In MB
                reserved_mem = torch.cuda.memory_reserved(device) / (1024**2) # In MB
                print(f"GPU Memory: {allocated_mem:.2f} MB Allocated / {reserved_mem:.2f} MB Reserved")
            # increase iteration count
            total_it += 1
            
            # enable training mode
            model.train()
            
            start = time.perf_counter()
            data = data.to(device)
            # In train.py -> main() -> training loop
            output, reg, _ = model(data)

# --- FIX: Adapt the output for the loss function ---
            if config["model"] == "neuralif":
    # The loss function expects (L, U). For our model M = L*L.T, so U = L.T.
    # The output from NeuralIF is L, so we create the tuple (L, L.T).
                output = (output, output.T)

            l = loss(
                output,
                data,
                config=config["loss"],
                c=config.get("c", reg),
                num_sketches=config["num_sketches"],
                pcg_steps=config["pcg_steps"],
                pcg_weight=config["pcg_weight"],
                normalized=config["normalized"],
                use_rademacher=config["use_rademacher"],
            )
            #  if reg:
            #  l = l + config["regularizer"] * reg
            
            l.backward()
            running_loss += l.item()
            
            # track the gradient norm
            if "gradient_clipping" in config and config["gradient_clipping"]:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clipping"])
            
            else:
                total_norm = 0.0
                
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
            
                grad_norm = total_norm ** 0.5 / config["batch_size"]
            
            # update network parameters
            optimizer.step()
            optimizer.zero_grad()
        
            # logger.log(l.item(), grad_norm, time.perf_counter() - start)
            
            # Do validation after 100 updates (to support big datasets)
            # convergence is expected to be pretty fast...
            if (total_it + 1) % 1000 == 0:
                
                # start with cg-checks after 5 iterations
                val_its = validate(model, validation_loader, solve=True,
                                    solver="gmres" if gmres else "cg")
                    
                # use scheduler
                # if config["scheduler"]:
                #    scheduler.step(val_loss)
                
                logger.log_val(None, val_its)
                
                # val_perf = val_cgits if val_cgits > 0 else val_loss
                val_perf = val_its
                
                if val_perf < best_val:
                    if config["save"]:
                        torch.save(model.state_dict(), f"{folder}/best_model.pt")
                    best_val = val_perf
        
        epoch_time = time.perf_counter() - start_epoch
        
        # save model every epoch for analysis...
        if config["save"]:
            torch.save(model.state_dict(), f"{folder}/model_epoch{epoch+1}.pt")
        
        print(f"Epoch {epoch+1} \t loss: {1/len(train_loader) * running_loss} \t time: {epoch_time}")
    
    # save fully trained model
    if config["save"]:
        logger.save_results()
        torch.save(model.to(torch.float).state_dict(), f"{folder}/final_model.pt")
    
    # Test the model
    # wandb.run.summary["validation_chol"] = best_val
    print()
    print("Best validation loss:", best_val)


def argparser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--device", type=int, required=False)
    parser.add_argument("--save", action='store_true')
    
    # Training parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--dataset", type=str, default="random")
    parser.add_argument("--loss", type=str, required=False)
    parser.add_argument("--gradient_clipping", type=float, default=1.0)

    parser.add_argument("--num_sketches", type=int, default=2,
                    help="number of sketch vectors in sketch_pcg loss")
    parser.add_argument("--pcg_steps", type=int, default=3,
                    help="how many CG iterations to unroll in the proxy")
    parser.add_argument("--pcg_weight", type=float, default=0.1,
                    help="weight of the CG-proxy term in the loss")
    parser.add_argument("--normalized", action="store_true",
                    help="normalize each sketch by ||A z||")
    parser.add_argument("--use_rademacher", action="store_true",
                    help="draw +/-1 sketches instead of Gaussian")
    parser.add_argument("--validation_dataset", type=str, default=None,
                    help="Path to the validation dataset directory.")
    parser.add_argument("--load_model_path", type=str, default=None, help="Path to a saved model state_dict to start training from.")
    parser.add_argument("--regularizer", type=float, default=0)
    parser.add_argument("--scheduler", action='store_true', default=False)
    
    # Model parameters
    parser.add_argument("--model", type=str, default="neuralif")
    
    parser.add_argument("--normalize", action='store_true', default=False)
    parser.add_argument("--latent_size", type=int, default=8)
    parser.add_argument("--message_passing_steps", type=int, default=3)
    parser.add_argument("--decode_nodes", action='store_true', default=False)
    parser.add_argument("--normalize_diag", action='store_true', default=False)
    parser.add_argument("--aggregate", nargs="*", type=str)
    parser.add_argument("--activation", type=str, default="relu")
    
    # NIF parameters
    parser.add_argument('--no-skip-connections', dest='skip_connections', action='store_false')
    parser.add_argument("--augment_nodes", action='store_true')
    parser.add_argument("--global_features", type=int, default=0)
    parser.add_argument("--edge_features", type=int, default=1)
    parser.add_argument("--graph_norm", action='store_true')
    parser.add_argument("--two_hop", action='store_true')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = argparser()
    
    if args.device is None:
        device = "cpu"
        print("Warning!! Using cpu only training")
        print("If you have a GPU use that with the command --device {id}")
        print()
    else:
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    
    if args.name is not None:
        folder = "results/" + args.name
    else:
        folder = folder = "results/" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    print(f"Using device: {device}")
    print("Using config: ")
    pprint.pprint(vars(args))
    print()
    
    # run experiments
    main(vars(args))

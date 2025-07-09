import copy
import os
import datetime
import argparse
import pprint
import torch
import torch_geometric
import time

from apps.data import get_dataloader, graph_to_matrix
from neuralif.utils import count_parameters, save_dict_to_file
from neuralif.logger import TrainResults
from neuralif.loss import loss
from neuralif.models import NeuralIF, PreCondNet, LearnedLU, NeuralPCG, NeuralIFWithRCM


@torch.no_grad()
def validate(model, validation_loader):
    """
    A fast and stable validation function that uses a 'sketched' loss
    to evaluate model performance without getting stuck on large graphs.
    """
    model.eval()
    
    total_loss = 0.0
    num_samples = 0
    device = next(model.parameters()).device # Get device from model

    for data in validation_loader:
        data = data.to(device)
        
        output, _, _ = model(data)
        
        # Prepare the model output tuple (L, L.T) for the loss function
        if isinstance(model, NeuralIF) or \
           (isinstance(model, torch.nn.DataParallel) and isinstance(model.module, NeuralIF)):
            output = (output, output.T)

        # Explicitly use the fast 'sketched' loss
        l = loss(output, data, config='sketched')
        
        total_loss += l.item()
        num_samples += 1
            
    avg_loss = total_loss / num_samples
    print(f"\nValidation Loss: {avg_loss:.4f}")
    return avg_loss

# ===================================================================

def main(config):
    folder = None
    if config["save"]:
        # Create a unique folder for the run inside the specified save directory
        folder_name = config.get("name", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        folder = os.path.join(config["save_dir"], folder_name)
        os.makedirs(folder, exist_ok=True)
        save_dict_to_file(config, os.path.join(folder, "config.json"))
        print(f"Results will be saved to: {os.path.abspath(folder)}")
    
    torch_geometric.seed_everything(config["seed"])
    
    device = torch.device(f"cuda:{config['device']}" if torch.cuda.is_available() and config.get("device") is not None else "cpu")
    print(f"Using device: {device}")
    
    model_args = {k: config[k] for k in ["latent_size", "message_passing_steps", "skip_connections",
                                         "augment_nodes", "global_features", "decode_nodes",
                                         "normalize_diag", "activation", "aggregate", "graph_norm",
                                         "two_hop", "edge_features"]
                  if k in config}
    
    # Model creation
    if config["model"] == "neuralif":
        model = NeuralIF(**model_args)
    elif config["model"] == "neuralif_rcm":
        print("--- Using NeuralIF model with RCM permutation ---")
        model = NeuralIFWithRCM(**model_args)
    else:
        # Add other model initializations here if needed
        raise NotImplementedError(f"Model {config['model']} not configured in this script.")
    
    if config["load_model_path"]:
        print(f"Loading model weights from: {config['load_model_path']}")
        model.load_state_dict(torch.load(config['load_model_path'], map_location=device))
    
    model.to(device)
    print(f"Number params in model: {count_parameters(model)}\n")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20)
    
    train_loader = get_dataloader(dataset_name=config["dataset"], batch_size=config["batch_size"], mode="train")
    # This line robustly determines the correct path for the validation set.
    validation_path = config["validation_dataset"] if config["validation_dataset"] else config["dataset"]

    validation_loader = get_dataloader(dataset_name=validation_path,
                                       batch_size=1,
                                       mode="val")
    
    logger = TrainResults(folder)
    best_val_loss = float('inf')
    total_it = 0
    
    for epoch in range(config["num_epochs"]):
        model.train()
        running_loss = 0.0
        start_epoch = time.perf_counter()
        
        for it, data in enumerate(train_loader):
            total_it += 1
            data = data.to(device)
            
            output, reg, _ = model(data)

            if config["model"] == "neuralif":
                output = (output, output.T)

            l = loss(
                output, data,
                config=config["loss"],
                num_sketches=config["num_sketches"],
                pcg_steps=config["pcg_steps"],
                pcg_weight=config["pcg_weight"],
                normalized=config["normalized"],
                use_rademacher=config["use_rademacher"],
            )
            
            if reg is not None and config.get("regularizer", 0) > 0:
                l = l + config["regularizer"] * reg
            
            l.backward()
            
            if config.get("gradient_clipping"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clipping"])
            
            optimizer.step()
            optimizer.zero_grad()
            running_loss += l.item()
            
            # --- Validation check ---
            if (total_it % 1000) == 0:
                val_loss = validate(model, validation_loader)
                
                if config["scheduler"]:
                    scheduler.step(val_loss)
                
                logger.log_val(val_loss, -1) 
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"New best validation loss: {best_val_loss:.4f}. Saving model.")
                    if config["save"]:
                        torch.save(model.state_dict(), f"{folder}/best_model.pt")
                
                model.train() # Set model back to training mode
        
        epoch_time = time.perf_counter() - start_epoch
        avg_epoch_loss = running_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}/{config['num_epochs']} \t Training Loss: {avg_epoch_loss:.4f} \t Time: {epoch_time:.2f}s")
        
        if config["save"]:
            torch.save(model.state_dict(), f"{folder}/model_epoch{epoch+1}.pt")
    
    print("\nTraining complete.")
    if config["save"]:
        print(f"Best validation loss: {best_val_loss:.4f}")
        logger.save_results()
        torch.save(model.state_dict(), f"{folder}/final_model.pt")

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
    parser.add_argument("--loss", type=str, required=True)
    parser.add_argument("--gradient_clipping", type=float, default=1.0)
    parser.add_argument("--num_sketches", type=int, default=2)
    parser.add_argument("--pcg_steps", type=int, default=3)
    parser.add_argument("--pcg_weight", type=float, default=0.1)
    parser.add_argument("--normalized", action="store_true")
    parser.add_argument("--use_rademacher", action="store_true")
    parser.add_argument("--validation_dataset", type=str, default=None)
    parser.add_argument("--load_model_path", type=str, default=None)
    parser.add_argument("--regularizer", type=float, default=0)
    parser.add_argument("--scheduler", action='store_true', default=False)
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Initial learning rate for the AdamW optimizer.")
    
    # Model parameters
    parser.add_argument("--model", type=str, default="neuralif")
    parser.add_argument("--latent_size", type=int, default=8)
    parser.add_argument("--message_passing_steps", type=int, default=3)
    parser.add_argument("--decode_nodes", action='store_true', default=False)
    parser.add_argument("--normalize_diag", action='store_true', default=False)
    parser.add_argument("--aggregate", nargs="*", type=str)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument('--no-skip-connections', dest='skip_connections', action='store_false', default=True)
    parser.add_argument("--augment_nodes", action='store_true')
    parser.add_argument("--global_features", type=int, default=0)
    parser.add_argument("--edge_features", type=int, default=1)
    parser.add_argument("--graph_norm", action='store_true')
    parser.add_argument("--two_hop", action='store_true')
    parser.add_argument("--save_dir", type=str, default="./results", help="Base directory to save results.")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = argparser()
    main(vars(args))

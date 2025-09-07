from .pretrain_model import DenoisePretrainModel, DenoisePretrainModelWithBlockEmbedding
from .affinity_predictor import AffinityPredictor
import torch

def create_model(args):
    # This function is simplified for PDBBind task only
    if args.task != 'PDBBind':
        raise ValueError(f"Only PDBBind task is supported, got {args.task}")
    
    add_params = {
        'num_affinity_pred_layers': args.num_pred_layers,
        'affinity_pred_dropout': args.pred_dropout,
        'affinity_pred_hidden_size': args.pred_hidden_size,
        'bottom_global_message_passing': args.bottom_global_message_passing,
        'global_message_passing': args.global_message_passing,
        'k_neighbors': args.k_neighbors,
        'dropout': args.dropout
    }
    
    # Set nonlinearity activation function
    if args.pred_nonlinearity == 'relu':
        add_params["nonlinearity"] = torch.nn.ReLU()
    elif args.pred_nonlinearity == 'gelu':
        add_params["nonlinearity"] = torch.nn.GELU()
    elif args.pred_nonlinearity == 'elu':
        add_params["nonlinearity"] = torch.nn.ELU()
    else:
        raise NotImplementedError(f"Nonlinearity {args.pred_nonlinearity} not implemented")
    
    # Create model from pretrained checkpoint or from scratch
    if args.pretrain_ckpt:
        print(f"Loading pretrain model from checkpoint {args.pretrain_ckpt}")
        add_params["partial_finetune"] = args.partial_finetune
        model = AffinityPredictor.load_from_pretrained(args.pretrain_ckpt, **add_params)
    elif args.pretrain_config:
        print(f"Loading pretrain model from config {args.pretrain_config}")
        model = AffinityPredictor.load_from_config_and_weights(args.pretrain_config, args.pretrain_weights, **add_params)
    else:
        model = AffinityPredictor(
            atom_hidden_size=args.atom_hidden_size,
            block_hidden_size=args.block_hidden_size,
            edge_size=args.edge_size,
            n_layers=args.n_layers,
            **add_params
        )
    return model

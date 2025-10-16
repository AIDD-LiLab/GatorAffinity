#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse
import torch
from torch.utils.data import DataLoader
import json
import numpy as np

from utils.logger import print_log
from utils.random_seed import setup_seed
from data.dataset import PDBBindBenchmark
import models
import trainers
from utils.nn_utils import count_parameters
from data.pdb_utils import VOCAB

import wandb

def parse():
    """
    Parse command line arguments for PDBBind benchmark training
    """
    parser = argparse.ArgumentParser(description='PDBBind Benchmark Training')
    
    # Data paths
    parser.add_argument('--train_set_path', type=str, default='LP-PDBbind/train.pkl', 
                        help='Path to training dataset')
    parser.add_argument('--valid_set_path', type=str, 
                        default='LP-PDBbind/val.pkl',
                        help='Path to validation dataset')
    
    
    # Learning rate parameters
    parser.add_argument('--lr', type=float, default=1e-6, help='Initial learning rate')
    parser.add_argument('--cos_lr', type=float, default=1e-7, help='Cosine annealing minimum learning rate')
    parser.add_argument('--final_lr', type=float, default=None, help='Final learning rate after training')
    parser.add_argument('--warmup_epochs', type=int, default=5, 
                        help='Number of warmup epochs before validation loss is used for early stopping')
    parser.add_argument('--warmup_start_lr', type=float, default=1e-6, help='Learning rate at warmup start')
    parser.add_argument('--warmup_end_lr', type=float, default=1e-4, help='Learning rate at warmup end')
    
    
    parser.add_argument('--bottom_global_message_passing', default=False)
    parser.add_argument('--global_message_passing', default=False)
    
    
    # Training parameters
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--max_epoch', type=int, default=200, help='Maximum number of training epochs')
    parser.add_argument('--grad_clip', type=float, default=None, help='Gradient clipping threshold')
    parser.add_argument('--batch_size', type=int, default=12, help='Training batch size')
    parser.add_argument('--valid_batch_size', type=int, default=12, help='Validation batch size')
    parser.add_argument('--patience', type=int, default=-1, 
                        help='Early stopping patience (-1 to disable)')
    parser.add_argument('--save_topk', type=int, default=-1, 
                        help='Save top-k checkpoints (-1 to save all improvements)')
    parser.add_argument('--save_dir', type=str, default='model_checkpoints', 
                        help='Directory to save model checkpoints and logs')
    
    # Data loading parameters
    parser.add_argument('--shuffle', default=True, action='store_true', help='Shuffle training data')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
    
    # Model architecture parameters
    parser.add_argument('--atom_hidden_size', type=int, default=32, 
                        help='Hidden dimension for atom features')
    parser.add_argument('--block_hidden_size', type=int, default=32, 
                        help='Hidden dimension for block features')
    parser.add_argument('--edge_size', type=int, default=32, 
                        help='Dimension of edge embeddings')
    parser.add_argument('--k_neighbors', type=int, default=8, 
                        help='Number of neighbors in KNN graph construction')
    parser.add_argument('--n_layers', type=int, default=4, 
                        help='Number of graph neural network layers')
    parser.add_argument('--partial_finetune', default=False, 
                        help='Enable partial fine-tuning of pretrained model')
    # Prediction head parameters
    parser.add_argument('--pred_dropout', type=float, default=0.0, 
                        help='Dropout rate in prediction head')
    parser.add_argument('--pred_nonlinearity', type=str, default='relu', 
                        choices=['relu', 'gelu', 'elu'], 
                        help='Activation function for prediction head')
    parser.add_argument('--num_pred_layers', type=int, default=3, 
                        help='Number of layers in prediction head')
    parser.add_argument('--pred_hidden_size', type=int, default=32, 
                        help='Hidden size for prediction head')
    parser.add_argument('--fragmentation_method', type=str, default='PS_300', 
                        help='Molecular fragmentation method for tokenization')
    # Pretrained model loading
    parser.add_argument('--pretrain_ckpt', type=str, default='./check_points/epoch6_step1148.ckpt', 
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--pretrain_config', type=str, default=None,
                        help='Path to pretrained model configuration')
    parser.add_argument('--pretrain_weights', type=str, default=None,
                        help='Path to pretrained model weights')
    
    # Device configuration
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], 
                        help='GPU IDs to use (-1 for CPU)')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")
    
    # Logging parameters
    parser.add_argument('--use_wandb', action="store_true", default=False, 
                        help='Enable Weights and Biases logging')
    parser.add_argument('--run_name', type=str, default="epoch40_step40549", 
                        help='Run name for logging')
    
    return parser.parse_args()




def main_pdbbind(args):
    """
    Main training function for PDBBind benchmark
    
    Args:
        args: Parsed command line arguments
    """
    # Set random seed for reproducibility
    setup_seed(args.seed)
    
    # Load vocabulary/tokenizer
    VOCAB.load_tokenizer(args.fragmentation_method if hasattr(args, 'fragmentation_method') else None)
    
    # Create model for PDBBind task
    args.task = 'PDBBind'  # Set task for model creation
    args.num_nodes = None  # PDBBind task doesn't require masking tokens
    model = models.create_model(args)
    
    # Load training and validation datasets
    print_log("Loading PDBBind datasets...")
    train_set = PDBBindBenchmark(args.train_set_path)
    
    if args.valid_set_path is not None:
        valid_set = PDBBindBenchmark(args.valid_set_path)
        print_log(f'Train set size: {len(train_set)}, Validation set size: {len(valid_set)}')
    else:
        valid_set = None
        print_log(f'Train set size: {len(train_set)}, No validation set')
    
    # Calculate steps per epoch for learning rate scheduling
    step_per_epoch = (len(train_set) + args.batch_size - 1) // args.batch_size
    
    # Configure training parameters
    config = trainers.TrainConfig(
        save_dir=args.save_dir,
        lr=args.lr,
        max_epoch=args.max_epoch,
        warmup_epochs=args.warmup_epochs,
        warmup_start_lr=args.warmup_start_lr,
        warmup_end_lr=args.warmup_end_lr,
        patience=args.patience,
        grad_clip=args.grad_clip,
        save_topk=args.save_topk,
    )
    
    # Add additional parameters for learning rate scheduling
    config.add_parameter(
        step_per_epoch=step_per_epoch,
        final_lr=args.final_lr if args.final_lr is not None else args.lr,
        cos_lr=args.cos_lr
    )
    
    # Set validation batch size
    if args.valid_batch_size is None:
        args.valid_batch_size = args.batch_size
    
    # Print model information
    print_log(f'Model created for PDBBind task')
    print_log(f'Number of parameters: {count_parameters(model) / 1e6:.2f} M')
    
    if args.pretrain_ckpt:
        print_log(f'Loaded pretrained checkpoint from {args.pretrain_ckpt}')
    
    # Create data loaders
    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        collate_fn=train_set.collate_fn,
        worker_init_fn=lambda x: np.random.seed(args.seed + x)
    )
    
    if valid_set is not None:
        valid_loader = DataLoader(
            valid_set, 
            batch_size=args.valid_batch_size,
            num_workers=args.num_workers,
            collate_fn=valid_set.collate_fn,
            shuffle=False
        )
    else:
        valid_loader = None
    
    # Create trainer for PDBBind task
    trainer = trainers.AffinityTrainer(model, train_loader, valid_loader, config)
    
    # Save configuration
    print_log(f"Saving model checkpoints to: {config.save_dir}")
    os.makedirs(config.save_dir, exist_ok=True)
    
    with open(os.path.join(config.save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Initialize Weights and Biases logging if enabled
    if args.use_wandb:
        wandb.init(
            entity="",
            dir=config.save_dir,
            settings=wandb.Settings(start_method="fork"),
            project=f"",
            name=args.run_name,
            config=vars(args),
        )
    
    # Start training
    trainer.train(
        device_ids=args.gpus,
        local_rank=args.local_rank,
        use_wandb=args.use_wandb,
        use_raytune=False
    )
    
    return trainer.topk_ckpt_map


if __name__ == '__main__':
    # Parse command line arguments
    args = parse()
    
    # Run PDBBind training
    main_pdbbind(args)

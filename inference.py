#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse
import torch
from tqdm import tqdm
import json
import numpy as np
import re
import pickle
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.logger import print_log
from utils.random_seed import setup_seed, SEED
from data.dataset import PDBBindBenchmark, PDBDataset
from data.pdb_utils import VOCAB
from models.prediction_model import PredictionModel
from models.pretrain_model import DenoisePretrainModel
from trainers.abs_trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='PDBBind prediction')
    
    # Model loading options
    parser.add_argument('--model_ckpt', type=str, default='check_points/epoch6_step1148.ckpt', help='path of the model ckpt to load')

    parser.add_argument('--test_set_path', type=str, default='test_data/test.pkl', help='path to test set')
    # Batch processing
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--seed', type=int, default=SEED)
    
    # Device
    parser.add_argument('--device', type=str, default='cuda', help='device to use')
    
    # Fragmentation
    parser.add_argument('--fragmentation_method', type=str, default='PS_300')
    
    return parser.parse_args()


def load_model(args):
    """Load model from checkpoint or config+weights"""
    if args.model_ckpt:
        # Load complete model checkpoint
        checkpoint = torch.load(args.model_ckpt, map_location='cpu',weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Need to reconstruct model from args
            with open(os.path.join(os.path.dirname(args.model_ckpt), 'args.json'), 'r') as f:
                train_args = json.load(f)
            # Convert dict to namespace
            from argparse import Namespace
            train_args = Namespace(**train_args)
            import models
            model = models.create_model(train_args)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Direct model object
            model = checkpoint
    else:
        raise ValueError("Must provide either --model_ckpt or both --model_config and --model_weights")

    
    return model


def predict_batch(model, batch, device):
    """Run prediction on a single batch"""
    batch = Trainer.to_device(batch, device)
    
    with torch.no_grad():
        
        loss, pred = model(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
            label=batch['label'],
            block_embeddings=batch.get('block_embeddings', None),
            block_embeddings0=batch.get('block_embeddings0', None),
            block_embeddings1=batch.get('block_embeddings1', None))
        
    return pred.cpu().numpy()


def calculate_ci(y_true, y_pred):
    """Calculate concordance index (CI)"""
    n = len(y_true)
    concordant_pairs = 0
    total_pairs = 0
    
    for i in range(n):
        for j in range(i+1, n):
            if y_true[i] != y_true[j]:
                total_pairs += 1
                # If the ranking is correct
                if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or \
                   (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                    concordant_pairs += 1
                # Handle ties in predictions
                elif y_pred[i] == y_pred[j]:
                    concordant_pairs += 0.5
    
    if total_pairs == 0:
        return 0.5
    
    return concordant_pairs / total_pairs


def evaluate_predictions(y_true, y_pred):
    """Calculate all evaluation metrics"""
    # RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAE
    mae = mean_absolute_error(y_true, y_pred)
    
    # Pearson correlation
    pearson_corr, pearson_p = stats.pearsonr(y_true, y_pred)
    
    # Spearman correlation
    spearman_corr, spearman_p = stats.spearmanr(y_true, y_pred)
    
    # Concordance Index
    ci = calculate_ci(y_true, y_pred)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'Pearson': pearson_corr,
        'Pearson_p': pearson_p,
        'Spearman': spearman_corr,
        'Spearman_p': spearman_p,
        'CI': ci
    }


def main(args):
    all_predictions = {}
    all_true_values = {}
    
    setup_seed(args.seed)
    VOCAB.load_tokenizer(args.fragmentation_method)
    
    # Load model
    print_log(f"Loading model...")
    model = load_model(args)
    model = model.to(args.device)
    model.eval()
    
    # Load test dataset
    print_log(f"Loading test set from {args.test_set_path}")
    dataset = PDBBindBenchmark(args.test_set_path)
    print_log(f'Test set size: {len(dataset)}')
    
    # Run predictions
    for idx in tqdm(range(0, len(dataset), args.batch_size), desc="Predicting", total=len(dataset)//args.batch_size+1):
        items = dataset.data[idx:min(idx+args.batch_size, len(dataset))]
        
        batch = PDBBindBenchmark.collate_fn(items)
        
        # Get predictions
        predictions = predict_batch(model, batch, args.device)
        
        for item, pred in zip(items, predictions):
            all_predictions[item["id"]] = pred.item()
            match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', str(item["label"]))
            all_true_values[item["id"]] = float(match.group()) if match else np.nan
    # Ensure same order for predictions and true values
    pdb_ids = sorted(all_predictions.keys())
    y_pred = np.array([all_predictions[pdb_id] for pdb_id in pdb_ids])
    y_true = np.array([all_true_values[pdb_id] for pdb_id in pdb_ids])
    
    # Calculate metrics
    print_log("\n=== Evaluation Results ===")
    metrics = evaluate_predictions(y_true, y_pred)
    
    print_log(f"RMSE: {metrics['RMSE']:.4f}")
    print_log(f"MAE: {metrics['MAE']:.4f}")
    print_log(f"Pearson Correlation: {metrics['Pearson']:.4f} (p-value: {metrics['Pearson_p']:.2e})")
    print_log(f"Spearman Correlation: {metrics['Spearman']:.4f} (p-value: {metrics['Spearman_p']:.2e})")
    print_log(f"Concordance Index (CI): {metrics['CI']:.4f}")
    
    # Print summary statistics
    print_log("\n=== Summary Statistics ===")
    print_log(f"Number of samples: {len(y_true)}")
    print_log(f"True values - Mean: {np.mean(y_true):.4f}, Std: {np.std(y_true):.4f}")
    print_log(f"Predictions - Mean: {np.mean(y_pred):.4f}, Std: {np.std(y_pred):.4f}")
    
    # Optional: Create a results dataframe for analysis (but don't save)
    results_df = pd.DataFrame({
        'PDB_ID': pdb_ids,
        'True_Affinity': y_true,
        'Predicted_Affinity': y_pred,
        'Error': y_pred - y_true,
        'Absolute_Error': np.abs(y_pred - y_true)
    })



if __name__ == '__main__':
    args = parse_args()
    main(args)
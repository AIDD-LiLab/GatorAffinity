import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean
import torch

from data.pdb_utils import VOCAB
from .pretrain_model import DenoisePretrainModel
from .ATOMICA.utils import batchify
from .prediction_model import PredictionModel, PredictionReturnValue


class AffinityPredictor(PredictionModel):
    """
    Neural network model for predicting protein-ligand binding affinity.
    
    This model uses a hierarchical graph neural network architecture with:
    - Bottom level: atom-level message passing
    - Top level: residue/block-level message passing
    - Energy prediction head for affinity estimation
    
    The model can optionally incorporate block embeddings for enhanced representations.
    """

    def __init__(self, num_affinity_pred_layers, nonlinearity, affinity_pred_dropout, affinity_pred_hidden_size,block_embedding_size=None, block_embedding0_size=None, block_embedding1_size=None,
                   **kwargs) -> None:
        """
        Initialize the AffinityPredictor model.
        
        Args:
            num_affinity_pred_layers (int): Number of layers in the affinity prediction head
            nonlinearity (nn.Module): Activation function (relu, gelu, or elu)
            affinity_pred_dropout (float): Dropout rate for affinity prediction layers
            affinity_pred_hidden_size (int): Hidden dimension for affinity prediction layers
            num_projector_layers (int): Number of layers in block embedding projector
            projector_hidden_size (int): Hidden dimension for projector layers
            projector_dropout (float): Dropout rate for projector layers
            block_embedding_size (int, optional): Size of block embeddings (same for all blocks)
            block_embedding0_size (int, optional): Size of block embeddings for segment 0
            block_embedding1_size (int, optional): Size of block embeddings for segment 1
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)

        # Convert nonlinearity to string for config saving
        self.nonlinearity = 'relu' if isinstance(nonlinearity, nn.ReLU) else 'gelu' if nonlinearity == nn.GELU else 'elu' if nonlinearity == nn.ELU else None
        
        # Store architecture parameters
        self.num_affinity_pred_layers = num_affinity_pred_layers
        self.affinity_pred_dropout = affinity_pred_dropout
        self.affinity_pred_hidden_size = affinity_pred_hidden_size

        self.block_embedding_size = block_embedding_size
        self.block_embedding0_size = block_embedding0_size
        self.block_embedding1_size = block_embedding1_size

        # Build energy prediction head (FFN that outputs scalar energy)
        layers = [nonlinearity, nn.Dropout(affinity_pred_dropout), nn.Linear(self.hidden_size, affinity_pred_hidden_size)]
        for _ in range(0, num_affinity_pred_layers-2):
            layers.extend([nonlinearity, nn.Dropout(affinity_pred_dropout), nn.Linear(affinity_pred_hidden_size, affinity_pred_hidden_size)])
        layers.extend([nonlinearity, nn.Dropout(affinity_pred_dropout), nn.Linear(affinity_pred_hidden_size, 1)])
        self.energy_ffn = nn.Sequential(*layers)

        # Option 1: Same block embedding for all blocks
        self.block_embedding_size = block_embedding_size
        if self.block_embedding_size:
            params = (nonlinearity, block_embedding_size)
            # Pre-encoder projectors and mixing layers
            block_projector, block_mixing = self.init_block_embedding(*params)
            self.pre_projector = nn.Sequential(*block_projector)
            self.pre_mixing_ffn = nn.Sequential(*block_mixing)
            # Post-encoder projectors and mixing layers
            block_projector, block_mixing = self.init_block_embedding(*params)
            self.post_projector = nn.Sequential(*block_projector)
            self.post_mixing_ffn = nn.Sequential(*block_mixing)

        # Option 2: Different block embeddings for segment 0 (protein) and segment 1 (ligand)
        self.block_embedding0_size = block_embedding0_size
        self.block_embedding1_size = block_embedding1_size
        if self.block_embedding0_size and self.block_embedding1_size:
            params0 = (nonlinearity, block_embedding0_size)
            params1 = (nonlinearity, block_embedding1_size)

            # Pre-encoder projectors for segment 0
            block_projector0, block_mixing0 = self.init_block_embedding(*params0)
            self.pre_projector0 = nn.Sequential(*block_projector0)
            self.pre_mixing_ffn0 = nn.Sequential(*block_mixing0)

            # Pre-encoder projectors for segment 1
            block_projector1, block_mixing1 = self.init_block_embedding(*params1)
            self.pre_projector1 = nn.Sequential(*block_projector1)
            self.pre_mixing_ffn1 = nn.Sequential(*block_mixing1)

            # Post-encoder projectors for segment 0
            block_projector0, block_mixing0 = self.init_block_embedding(*params0)
            self.post_projector0 = nn.Sequential(*block_projector0)
            self.post_mixing_ffn0 = nn.Sequential(*block_mixing0)

            # Post-encoder projectors for segment 1
            block_projector1, block_mixing1 = self.init_block_embedding(*params1)
            self.post_projector1 = nn.Sequential(*block_projector1)
            self.post_mixing_ffn1 = nn.Sequential(*block_mixing1)

        # Disable attention pooling as it's not used for affinity prediction
        self.attention_pooling.requires_grad_(requires_grad=False)
    
    def init_block_embedding(self, nonlinearity: nn.Module, block_embedding_size: int, projector_dropout: float, 
                           projector_hidden_size: int, num_projector_layers: int):
        """
        Initialize projector and mixing layers for block embeddings.
        
        Args:
            nonlinearity: Activation function
            block_embedding_size: Input dimension of block embeddings
            projector_dropout: Dropout rate
            projector_hidden_size: Hidden dimension
            num_projector_layers: Number of layers
            
        Returns:
            tuple: (projector_layers, mixing_layers) - Lists of layers for projection and mixing
        """
        # Projector: transforms block embeddings to model hidden size
        projector_layers = [nonlinearity, nn.Dropout(projector_dropout), nn.Linear(block_embedding_size, projector_hidden_size)]
        for _ in range(0, num_projector_layers-2):
            projector_layers.extend([nonlinearity, nn.Dropout(projector_dropout), nn.Linear(projector_hidden_size, projector_hidden_size)])
        projector_layers.extend([nonlinearity, nn.Dropout(projector_dropout), nn.Linear(projector_hidden_size, self.hidden_size)])

        # Mixing: combines original features with projected block embeddings
        mixing_layers = [nonlinearity, nn.Dropout(projector_dropout), nn.Linear(2*self.hidden_size, 2*self.hidden_size)]
        for _ in range(0, num_projector_layers-2):
            mixing_layers.extend([nonlinearity, nn.Dropout(projector_dropout), nn.Linear(2*self.hidden_size, 2*self.hidden_size)])
        mixing_layers.extend([nonlinearity, nn.Dropout(projector_dropout), nn.Linear(2*self.hidden_size, self.hidden_size)])
        
        return projector_layers, mixing_layers

    @classmethod
    def _load_from_pretrained(cls, pretrained_model, **kwargs):
        """
        Load affinity predictor from a pretrained model.
        
        Args:
            pretrained_model: Pretrained model instance
            **kwargs: Additional configuration parameters
            
        Returns:
            AffinityPredictor: New model initialized with pretrained weights
        """
        # Check for k_neighbors mismatch
        if pretrained_model.k_neighbors != kwargs.get('k_neighbors', pretrained_model.k_neighbors):
            print(f"Warning: pretrained model k_neighbors={pretrained_model.k_neighbors}, new model k_neighbors={kwargs.get('k_neighbors')}")
        
        # Create new model with combined pretrained and new parameters
        model = cls(
            atom_hidden_size=pretrained_model.atom_hidden_size,
            block_hidden_size=pretrained_model.hidden_size,
            edge_size=pretrained_model.edge_size,
            k_neighbors=kwargs.get('k_neighbors', pretrained_model.k_neighbors),
            n_layers=pretrained_model.n_layers,
            dropout=kwargs.get('dropout', pretrained_model.dropout), # for backward compatibility
            bottom_global_message_passing=kwargs.get('bottom_global_message_passing', pretrained_model.bottom_global_message_passing),
            global_message_passing=kwargs['global_message_passing'],
            nonlinearity=kwargs['nonlinearity'],
            num_affinity_pred_layers=kwargs['num_affinity_pred_layers'],
            affinity_pred_dropout=kwargs['affinity_pred_dropout'],
            affinity_pred_hidden_size=kwargs['affinity_pred_hidden_size']
        )
        
        # Print loaded model configuration
        print(f"""Pretrained model params: hidden_size={model.hidden_size},
               edge_size={model.edge_size}, k_neighbors={model.k_neighbors}, 
               n_layers={model.n_layers}, bottom_global_message_passing={model.bottom_global_message_passing},
               global_message_passing={model.global_message_passing} 
               """)
        
        # Ensure no noise is applied in prediction model
        assert not any([model.atom_noise, model.translation_noise, model.rotation_noise, model.torsion_noise]), "prediction model no noise"
        
        # Load pretrained weights (strict=False allows for new layers)
        if pretrained_model.state_dict() is not None:
            model.load_state_dict(pretrained_model.state_dict(), strict=False)

        # Handle partial finetuning if requested
        partial_finetune = kwargs.get('partial_finetune', False)
        if partial_finetune:
            model.requires_grad_(requires_grad=False)

        # Handle edge embedder training for global message passing
        if pretrained_model.global_message_passing is False and model.global_message_passing is True:
            model.edge_embedding_top.requires_grad_(requires_grad=True)
            print("Warning: global_message_passing is True in the new model but False in the pretrain model, training edge_embedders in the model")
        
        if pretrained_model.bottom_global_message_passing is False and model.bottom_global_message_passing is True:
            model.edge_embedding_bottom.requires_grad_(requires_grad=True)
            print("Warning: bottom_global_message_passing is True in the new model but False in the pretrain model, training edge_embedders in the model")
        
        # Disable attention pooling (not used in affinity prediction)
        model.attention_pooling.requires_grad_(requires_grad=False)
        
        # Enable energy FFN for partial finetuning
        if partial_finetune:
            model.energy_ffn.requires_grad_(requires_grad=True)
            
        return model
    
    def get_config(self):
        """
        Get model configuration dictionary.
        
        Returns:
            dict: Configuration parameters for model reconstruction
        """
        config_dict = super().get_config()
        config_dict.update({
            'nonlinearity': self.nonlinearity,
            'num_affinity_pred_layers': self.num_affinity_pred_layers,
            'affinity_pred_dropout': self.affinity_pred_dropout,
            'affinity_pred_hidden_size': self.affinity_pred_hidden_size
        })
        return config_dict

    def forward(self, Z, B, A, block_lengths, lengths, segment_ids, label, 
                block_embeddings, block_embeddings0, block_embeddings1) -> PredictionReturnValue:
        """
        Forward pass for affinity prediction.
        
        Args:
            Z (Tensor): Atom coordinates [N_atoms, 3]
            B (Tensor): Block (residue) types [N_blocks]
            A (Tensor): Atom types [N_atoms]
            block_lengths (Tensor): Number of atoms in each block [N_blocks]
            lengths (Tensor): Number of blocks in each molecule [batch_size]
            segment_ids (Tensor): Segment IDs (0=protein, 1=ligand) [N_blocks]
            label (Tensor): Ground truth affinity values [batch_size]
            
        Returns:
            PredictionReturnValue: (loss, predictions)
        """
        # Compute batch and block indices
        with torch.no_grad():
            # batch_id: which molecule each block belongs to
            batch_id = torch.zeros_like(segment_ids)  # [Nb]
            batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
            batch_id.cumsum_(dim=0)  # [Nb], item idx in the batch

            # block_id: which block each atom belongs to
            block_id = torch.zeros_like(A) # [Nu]
            block_id[torch.cumsum(block_lengths, dim=0)[:-1]] = 1
            block_id.cumsum_(dim=0)  # [Nu], block (residue) id of each unit (atom)

            # Map block-level info to atom level
            bottom_batch_id = batch_id[block_id]  # [Nu]
            bottom_B = B[block_id]  # [Nu]
            bottom_segment_ids = segment_ids[block_id]  # [Nu]
            bottom_block_id = torch.arange(0, len(block_id), device=block_id.device)  #[Nu]

        # Initial embeddings
        bottom_H_0 = self.block_embedding.atom_embedding(A)  # Atom embeddings
        top_H_0 = self.block_embedding.block_embedding(B)    # Block embeddings
        
        # Incorporate external block embeddings if provided
        if self.block_embedding_size:
            # Same embedding for all blocks
            block_embeddings_all = self.pre_projector(block_embeddings)
            top_H_0 = self.pre_mixing_ffn(torch.cat([top_H_0, block_embeddings_all], dim=-1))
        elif self.block_embedding0_size and self.block_embedding1_size:
            # Different embeddings for protein and ligand segments
            block_embeddings_segment0 = self.pre_projector0(block_embeddings0)
            block_embeddings_segment1 = self.pre_projector1(block_embeddings1)
            top_H_0_segment0 = self.pre_mixing_ffn0(torch.cat([top_H_0[segment_ids==0], block_embeddings_segment0], dim=-1))
            top_H_0_segment1 = self.pre_mixing_ffn1(torch.cat([top_H_0[segment_ids==1], block_embeddings_segment1], dim=-1))
            top_H_0 = torch.cat([top_H_0_segment0, top_H_0_segment1], dim=0)

        # Bottom level (atom) message passing
        edges, edge_attr = self.get_edges(bottom_B, bottom_batch_id, bottom_segment_ids, 
                                          Z, bottom_block_id, self.bottom_global_message_passing, 
                                          top=False)
        bottom_block_repr = self.encoder(
            bottom_H_0, Z, bottom_batch_id, None, edges, edge_attr, 
        )
        
        # Top level (block/residue) message passing
        top_Z = scatter_mean(Z, block_id, dim=0)  # Average atom positions to get block centers [Nb, n_channel, 3]
        top_block_id = torch.arange(0, len(batch_id), device=batch_id.device)
        edges, edge_attr = self.get_edges(B, batch_id, segment_ids, top_Z, top_block_id, 
                                          self.global_message_passing, top=True)
        
        # Aggregate atom representations to block level
        if self.bottom_global_message_passing:
            batched_bottom_block_repr, _ = batchify(bottom_block_repr, block_id)
        else:
            # Exclude global atoms from aggregation
            atom_mask = A != VOCAB.get_atom_global_idx()
            batched_bottom_block_repr, _ = batchify(bottom_block_repr[atom_mask], block_id[atom_mask])
        
        # Attention from blocks to atoms
        block_repr_from_bottom = self.atom_block_attn(top_H_0.unsqueeze(1), batched_bottom_block_repr)
        top_H_0 = top_H_0 + block_repr_from_bottom.squeeze(1)
        top_H_0 = self.atom_block_attn_norm(top_H_0)

        # Top-level encoding
        top_block_id = torch.arange(0, len(batch_id), device=batch_id.device)
        block_repr = self.top_encoder(top_H_0, top_Z, batch_id, None, edges, edge_attr)

        # Incorporate block embeddings after encoding
        if self.block_embedding_size:
            block_embeddings_all = self.post_projector(block_embeddings)
            block_repr = self.post_mixing_ffn(torch.cat([block_repr, block_embeddings_all], dim=-1))
        elif self.block_embedding0_size and self.block_embedding1_size:
            block_embeddings_segment0 = self.post_projector0(block_embeddings0)
            block_embeddings_segment1 = self.post_projector1(block_embeddings1)
            block_repr_segment0 = self.post_mixing_ffn0(torch.cat([block_repr[segment_ids==0], block_embeddings_segment0], dim=-1))
            block_repr_segment1 = self.post_mixing_ffn1(torch.cat([block_repr[segment_ids==1], block_embeddings_segment1], dim=-1))
            block_repr = torch.cat([block_repr_segment0, block_repr_segment1], dim=0)

        # Predict energy contribution from each block
        block_energy = self.energy_ffn(block_repr).squeeze(-1)
        
        # Ignore global blocks if global message passing is disabled
        if not self.global_message_passing:
            block_energy[B == self.global_block_id] = 0
        
        # Sum block energies to get molecule-level affinity
        pred_energy = scatter_sum(block_energy, batch_id)
        
        # Compute MSE loss
        return F.mse_loss(pred_energy, label), pred_energy  

    def infer(self, batch):
        """
        Inference method for affinity prediction.
        
        Args:
            batch (dict): Batch dictionary containing model inputs
            
        Returns:
            Tensor: Predicted affinity values
        """
        self.eval()
        loss, pred_energy = self.forward(
            Z=batch['X'],
            B=batch['B'],
            A=batch['A'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
            label=batch['label'],
            block_embeddings=batch.get('block_embeddings', None),
            block_embeddings0=batch.get('block_embeddings0', None),
            block_embeddings1=batch.get('block_embeddings1', None),
        )
        return pred_energy
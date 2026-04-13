import torch
from torch import nn
import math
from typing import Dict


class TransformerBlock(nn.Module):
    """
    TransformerBlock is a single block of the transformer architecture, which includes
    multi-head self-attention, layer normalization, and a feed-forward network.

    Args:
        n_heads (int): Number of attention heads.
        embedding_dimension (int): Dimension of the input embeddings.
        ffn_hidden_dimension (int): Dimension of the hidden layer in the feed-forward network.
        dropout (float): Dropout rate to apply after attention and feed-forward layers.
    """
    def __init__(self, n_heads, embedding_dimension, ffn_hidden_dimension, dropout):
        super(TransformerBlock, self).__init__()
        # Multihead Attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dimension,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Normalization layers
        self.norm_attn = nn.LayerNorm(embedding_dimension)
        self.norm_ffn = nn.LayerNorm(embedding_dimension)
        # Dropouts
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)
        # FeedForward layer
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dimension, ffn_hidden_dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dimension, embedding_dimension),
        )

    def forward(self, x, attn_mask=None):
        attn_x, _ = self.multihead_attn(
            query=x, key=x, value=x, key_padding_mask=attn_mask, need_weights=False
        )
        x = self.norm_attn(self.dropout_attn(attn_x))
        ffn_x = self.ffn(x)
        output = self.norm_ffn(self.dropout_ffn(ffn_x))
        return output
 

class MultifidelityTransformer(nn.Module):
    """Multifidelity Transformer model that processes multiple levels of fidelity data
    and combines them using a transformer architecture.

    Args:
        levels_dim (Dict[str, int]): Dictionary mapping level names to their dimensions.
        output_dim (int): Dimension of the output.
        embedding_dim (int): Dimension of the embeddings for each level.
        parameters_dim (int): Number of model parameters (e.g., time and control parameters) 
        n_heads (int): Number of attention heads in the transformer.
        n_masks (int, optional): Number of mask combinations. Defaults to None. TODO: explain
        dropout (float, optional): Dropout rate for the model. Defaults to 0.2
        n_transformer_blocks (int, optional): Number of transformer blocks to use. Defaults to 1.
        spatial_encoders_dim (Dict[str, int], optional): Dictionary mapping level names to their
            spatial encoder dimensions. Defaults to None.
    """ 

    def __init__(
        self,
        levels_dim: Dict[str, int],
        output_dim: int,
        embedding_dim: int,
        parameters_dim: int,
        n_heads: int,
        n_masks=None,
        dropout: float = 0.2,
        n_transformer_blocks=1,
        spatial_encoders_dim: Dict[str, int]=None
    ):
        super().__init__()
        # Save the number of levels and the ids of the mask combinations
        self.n_levels = len(levels_dim)
        self.n_masks = n_masks
        self.embedding_dim = embedding_dim

        # Set the dimension of the inputs
        self.parameters_dim = parameters_dim
        self.levels_dim = levels_dim
        self.output_dim = output_dim
        self.spatial_encoders_dim = spatial_encoders_dim

        # Build the Spatial Encoders for the second and third fidelity level
        if spatial_encoders_dim:
            self.spatial_encoders = nn.ModuleList(
                [
                    nn.Linear(in_features=levels_dim[level], out_features=dim)
                    for level, dim in self.spatial_encoders_dim.items()
                ]
            )
            # Adjust the input size for the RNN encoders.
            for level, dim in self.spatial_encoders_dim.items():
                levels_dim[level] = dim

        # Build the first layer encoders for each fidelity level
        self.levels_encoders = nn.ModuleList(
            [
                nn.LSTM(input_size=dim, hidden_size=self.embedding_dim, batch_first=True)
                for _, dim in levels_dim.items()
            ]
        )
        self.dropouts_encoders = nn.ModuleList(
            [nn.Dropout(p=dropout) for _ in range(self.n_levels)]
        )

        # Embedding Layer for the masks (inject knowledge on how many levels are available)
        self.mask_embeddings = None
        if n_masks is not None:
            self.mask_embeddings = nn.Embedding(
                num_embeddings=self.n_levels * self.n_masks,
                embedding_dim=self.embedding_dim,
            )

        # Transformer blocks to merge the information carried by each level
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                n_heads=n_heads,
                embedding_dimension=embedding_dim,
                ffn_hidden_dimension=embedding_dim * 4,
                dropout=dropout) for _ in range(n_transformer_blocks)
            ]
        )

        # Final projection to the output space
        self.output_projection = nn.Linear(
            embedding_dim + self.parameters_dim, output_dim
        )

    def forward(self, x):
        # Get the sequence length of the batch :
        # an input level from x has shape (batch_size, seq_length, level_dim)
        seq_length = x["level_0"].size(1)

        # BLOCK 1: Encode each level
        # Spatial Encoding for the corresponding levels 
        if self.spatial_encoders_dim:
            for k, level in enumerate(self.spatial_encoders_dim.keys()):
                x[level] = self.spatial_encoders[k](x[level])
        # Time encoding with RNN        
        encoded_levels = tuple(
            encoder(x[f"level_{level}"])[0]
            for level, encoder in enumerate(self.levels_encoders, 1)
        )  # Tuple contains self.n_levels tensors of shape (batch_size, seq_len, embedding_dim)
        encoded_sequence = torch.cat(
            [
                dropout(
                    encoded_levels[level].contiguous().view(-1, 1, self.embedding_dim)
                )
                for level, dropout in enumerate(self.dropouts_encoders)
            ],
            dim=1,
        )  # Tensor of shape (batch_size * seq_len, self.n_levels, embedding_dim)

        # BLOCK 2: Transformer to mix all the fidelity levels.
        # Assemble the masks for each timestep in the sequence.
        levels_mask = x["mask"].repeat_interleave(
            seq_length, dim=0
        )  # (batch_size * seq_len, embedding_dim)

        # If we are using mask embeddings, add them.
        if self.mask_embeddings is not None:
            ids = x["input_ids"].repeat_interleave(
                seq_length, dim=0
            )  # (batch_size * seq_len, n_levels, embedding_dim)
            encoded_sequence += self.mask_embeddings(ids)

        # Process with the transformer blocks each embedding.
        for transformer_block in self.transformer_blocks:
            encoded_sequence = transformer_block(
                x=encoded_sequence, attn_mask=levels_mask
            )  # (batch_size * seq_len, n_levels, embedding_dim)

        # BLOCK 3: Pooling.
        # Set to 0 all the embeddings of the levels that we don't have
        transformer_output = encoded_sequence.masked_fill(
            mask=levels_mask.unsqueeze(-1), value=0.0
        )
        # Average the embeddings for the levels that we have
        final_embedding = transformer_output.sum(dim=-2) / (levels_mask == False).sum(
            dim=-1
        ).unsqueeze(
            -1
        )  # (batch_size * seq_len, embedding_dim)

        # BLOCK 4: Decoding.
        # Now we use the Multifidelity embedding to decode the output
        output = (
            self.output_projection(
                torch.cat(
                    (
                        final_embedding,
                        x["level_0"].contiguous().view(-1, self.parameters_dim),
                    ),
                    dim=1,
                )
            )
            .contiguous()
            .view(-1, seq_length, self.output_dim)
        )
        return output
        
class BaselineModel(nn.Module):
    def __init__(self, levels_dim: Dict[str, int], embedding_dim: int, output_dim: int, 
                 n_hidden: int, dropout: float=0.1, spatial_encoders_dim: Dict[str, int]=None):
        super().__init__()

        # Save the number of levels 
        self.n_levels = len(levels_dim)
        self.embedding_dim = embedding_dim
        self.n_hidden = n_hidden

        # Set the dimension of the inputs
        self.parameters_dim = 2  # TODO: Hardcoded for now, need to change it
        self.levels_dim = levels_dim
        self.output_dim = output_dim
        self.spatial_encoders_dim = spatial_encoders_dim
        
        # Build the Spatial Encoders for the second and third fidelity level
        if spatial_encoders_dim:
            self.spatial_encoders = nn.ModuleList(
                [
                    nn.Linear(in_features=levels_dim[level], out_features=dim)
                    for level, dim in self.spatial_encoders_dim.items()
                ]
            )
            # Adjust the input size for the RNN encoders.
            for level, dim in self.spatial_encoders_dim.items():
                levels_dim[level] = dim

        # Build the first layer encoders for each fidelity level
        self.levels_encoders = nn.ModuleList(
            [
                nn.LSTM(input_size=dim, hidden_size=self.embedding_dim, batch_first=True)
                for _, dim in levels_dim.items()
            ]
        )

        # Dropouts
        self.dropouts_encoders = nn.ModuleList(
            [nn.Dropout(p=dropout) for _ in range(self.n_levels)]
        )

        # Define a fully connected layer to combine the outputs of the GRU layers
        self.fc_layers_mixing = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(embedding_dim * (len(self.levels_dim) - i), embedding_dim * (len(self.levels_dim) - i - 1)),
                nn.LayerNorm(embedding_dim * (len(self.levels_dim) - i - 1)),
                nn.ReLU(),
                nn.Dropout(p=dropout)) for i in range(len(self.levels_dim) - 1)]
        )

        self.fc_hidden_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(embedding_dim + self.parameters_dim, embedding_dim + self.parameters_dim),
                nn.LayerNorm(embedding_dim + self.parameters_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout)) for _ in range(self.n_hidden)]
        )
        self.fc_output = nn.Linear(embedding_dim + self.parameters_dim, output_dim)  

    def forward(self, x):
        # Get the sequence length of the batch :
        # an input level from x has shape (batch_size, seq_length, level_dim)
        seq_length = x["level_0"].size(1)

        # BLOCK 1: Encode each level
        # Spatial Encoding for the corresponding levels 
        if self.spatial_encoders_dim:
            for k, level in enumerate(self.spatial_encoders_dim.keys()):
                x[level] = self.spatial_encoders[k](x[level])
        # Time encoding with RNN        
        encoded_levels = tuple(
            encoder(x[f"level_{level}"])[0]
            for level, encoder in enumerate(self.levels_encoders, 1)
        )  # Tuple contains self.n_levels tensors of shape (batch_size, seq_len, embedding_dim)

        encoded_sequence = torch.cat(
            [
                dropout(
                    encoded_levels[level].contiguous().view(-1, self.embedding_dim)
                )
                for level, dropout in enumerate(self.dropouts_encoders)
            ],
            dim=1,
        )  # Tensor of shape (batch_size * seq_len, self.n_levels, embedding_dim)
        
        # Pass through fully connected layers
        for i, fc_mix in enumerate(self.fc_layers_mixing):
            encoded_sequence = fc_mix(encoded_sequence)

        # Concat the parameters
        combined = torch.cat(
                    (
                        encoded_sequence,
                        x["level_0"].contiguous().view(-1, self.parameters_dim),
                    ),
                    dim=1,
                ) # Tensor of shape (batch_size * seq_len, embedding_dim * self.n_levels + self.parameters_dim)
        
        for i, fc_hidden in enumerate(self.fc_hidden_layers):
            combined = fc_hidden(combined)

        out = self.fc_output(combined)
        
        return out.contiguous().view(-1, seq_length, self.output_dim)

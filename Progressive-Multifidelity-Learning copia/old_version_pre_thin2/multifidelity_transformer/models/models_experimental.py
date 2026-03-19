import torch
from torch import nn
import math
from typing import Dict, Optional, Tuple


class TransformerBlock(nn.Module):
    """
    A single transformer block with multi-head self-attention and feed-forward network.

    Args:
        n_heads: Number of attention heads
        embedding_dim: Dimension of the embeddings
        ffn_hidden_dim: Dimension of the hidden layer in the feed-forward network
        dropout: Dropout rate to apply after attention and feed-forward layers
    """
    def __init__(self, n_heads : int, embedding_dim : int, ffn_hidden_dim : int, dropout : float = 0.1):
        super().__init__()

        # Multihead Attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Normalization layers
        self.norm_attn = nn.LayerNorm(embedding_dim)
        self.norm_ffn = nn.LayerNorm(embedding_dim)
        # Dropouts
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)
        # FeedForward layer
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attn_x, _ = self.multihead_attn(
            query=x, key=x, value=x, key_padding_mask=attn_mask, need_weights=False
        )
        x = self.norm_attn(x + self.dropout_attn(attn_x))

        # Feed-forward network
        ffn_x = self.ffn(x)
        output = self.norm_ffn(x + self.dropout_ffn(ffn_x))
        return output

class EncoderFactory:
    """ Factory for creating spatial encoders. """
    
    @staticmethod
    def create_encoder(encoder_type: str, input_dim: int, output_dim: int) -> nn.Module:
        """ Create an encoder based on the specified type. """
        #prima era solo linear(diminp, dimout)
        if encoder_type == "linear":
            # Usiamo nn.Sequential per impacchettare i layer in un singolo modulo
            return nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),  # Creiamo un'istanza del layer ReLU
                nn.Linear(128, output_dim)
            )
        
        elif encoder_type == "conv1d":
            # Questo era già corretto
            return nn.Conv1d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, padding=1)
        
        else:
            raise ValueError(f"Encoder type {encoder_type} not recognized.")
        
class MultifidelityTransformer(nn.Module):
    """ 
    Multifidelity Transformer for processing multiple levels of fidelity data.
    
    Args:
        levels_dim: Dictionary mapping fidelity level names to their input dimensions
        output_dim: Output dimension
        embedding_dim: Embedding dimension (same for all levels)
        parameters_dim: Number of model parameters (e.g., time and control parameters)
        n_heads: Number of attention heads
        n_masks: Number of mask combinations (optional) TODO: explain
        dropout: Dropout rate
        n_transformer_blocks: Number of transformer blocks
        spatial_encoders_config: Configuration for spatial encoders #TODO: add more options
    """ 

    def __init__(
        self,
        levels_dim: Dict[str, int],
        output_dim: int,
        embedding_dim: int,
        parameters_dim: int,
        n_heads: int,
        n_masks: Optional[int] = None,
        dropout: float = 0.1,
        n_transformer_blocks: int = 1,
        spatial_encoders_dim: Optional[Dict[str, int]] = None
    ):
        super().__init__()

        # Store configuration
        self.levels_dim = levels_dim
        self.n_levels = len(levels_dim)
        self.embedding_dim = embedding_dim
        self.parameters_dim = parameters_dim
        self.output_dim = output_dim
        self.n_masks = n_masks

        # Process spatial encoder configuration
        self.spatial_encoders_dim = spatial_encoders_dim if spatial_encoders_dim else {}

        # Build model components
        self.spatial_encoders = self._build_spatial_encoders()
        self.temporal_encoders = self._build_temporal_encoders(dropout)
        self.transformer_blocks = self._build_transformer_blocks(n_heads, n_transformer_blocks, dropout)
        self.decoder = self._build_decoder(dropout)

        # Optional mask embeddings
        self.mask_embeddings= self._build_mask_embeddings() if n_masks else None

    def _build_spatial_encoders(self) -> nn.ModuleDict:
        """Build spatial encoders for configured level."""
        encoders = nn.ModuleDict()
        
        for level_name, dim in self.spatial_encoders_dim.items():
            if level_name not in self.levels_dim:
                raise ValueError(f"Level {level_name} not found in levels_dim.")
        
            encoder_type = "linear"#config.get("type", "linear")
            output_dim = dim

            encoder = EncoderFactory.create_encoder(
                encoder_type,
                self.levels_dim[level_name],
                output_dim
            )
            
            encoders[level_name] = encoder
            # Update dimension for temporal encoders
            self.levels_dim[level_name] = output_dim

        return encoders

    def _build_temporal_encoders(self, dropout: float) -> nn.ModuleList:
        """Build temporal encoders for each fidelity level."""
        encoders = nn.ModuleDict()

        for level_name, dim in self.levels_dim.items():
            encoders[level_name] = nn.GRU(
                input_size=dim,
                hidden_size=self.embedding_dim,
                batch_first=True,
                dropout=dropout if dropout > 0 else 0.0
            )
        
        return encoders

    def _build_transformer_blocks(self, n_heads: int, n_transformer_blocks: int, dropout: float) -> nn.ModuleList:
        """Build transformer blocks to merge information from all levels."""
        transformer_blocks = nn.ModuleList()

        for _ in range(n_transformer_blocks):
            transformer_block = TransformerBlock(
                n_heads=n_heads,
                embedding_dim=self.embedding_dim,
                ffn_hidden_dim=self.embedding_dim, #TODO: embedding_dim*4  # Typically 4 times the embedding dimension
                dropout=dropout
            )
            transformer_blocks.append(transformer_block)
        
        return transformer_blocks

    def _build_decoder(self, dropout: float) -> nn.Sequential:
        """
        Build the decoder.
        SOLUTION 1 (Deep MLP): Instead of projecting directly 64 -> 60000,
        we do 64 -> 2048 -> ReLU -> 60000.
        """
        
        # Intermediate dimension to unpack information (Expansion Layer)
        # Can be in config, but 1024 or 2048 is standard for embedding=64
        hidden_decoder_dim = 256  # 512 has more parameters but is slower

        return nn.Sequential(
            # 0. The Decoder GRU (remains the same)
            nn.GRU(
                input_size=self.embedding_dim + self.parameters_dim,
                hidden_size=self.embedding_dim,
                batch_first=True,
                num_layers=1,
                dropout=dropout if dropout > 0 else 0.0
            ),
            
            # 1. The Projector (Now it's an MLP, not a single Linear)
            nn.Sequential(
                nn.Linear(self.embedding_dim, hidden_decoder_dim),
                nn.LayerNorm(hidden_decoder_dim),  # Stabilizes training
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_decoder_dim, self.output_dim)
            )
        )
    
    def _build_mask_embeddings(self) -> nn.Embedding:
        """Build embedding layer for masks."""
        return nn.Embedding(
            num_embeddings=self.n_levels * self.n_masks,
            embedding_dim=self.embedding_dim,
        )
    
    def _encode_spatial_features(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply spatial encoding to configured levels."""
        encoded = x.copy()
        
        for level_name, encoder in self.spatial_encoders.items():
            if level_name in encoded:
                encoded[level_name] = encoder(encoded[level_name])
        
        return encoded

    def _encode_temporal_features(self, x: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, int]:
        """Apply temporal encoding using GRU."""
        # Get sequence length from first level
        seq_length = next(iter(x.values())).size(1)
        
        # Encode each level
        encoded_levels = []
        for level_name, encoder in self.temporal_encoders.items():
            if level_name in x:
                encoded_output, _ = encoder(x[level_name])
                # Reshape for concatenation: (batch_size * seq_len, 1, embedding_dim)
                reshaped = encoded_output.contiguous().view(-1, 1, self.embedding_dim)
                encoded_levels.append(reshaped)
        
        # Concatenate all levels: (batch_size * seq_len, n_levels, embedding_dim)
        encoded_sequence = torch.cat(encoded_levels, dim=1)
        
        return encoded_sequence, seq_length
    
    def _apply_transformer_blocks(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply transformer blocks for information fusion."""
        for block in self.transformer_blocks:
            x = block(x, attn_mask=mask)
        return x
    
    def _pool_embeddings(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Pool embeddings across fidelity levels."""
        # Set masked positions to 0
        x_masked = x.masked_fill(mask.unsqueeze(-1), 0.0)
        
        # Average over available levels
        valid_count = (~mask).sum(dim=-1, keepdim=True)
        pooled = x_masked.sum(dim=-2) / valid_count.clamp(min=1)
        
        return pooled

    def forward(self, x : Dict[str, torch.Tensor]) -> torch.Tensor:
            """ Forward pass through the multifidelity transformer model. """
            # Step 1: Spatial encoding
            x_encoded =  self._encode_spatial_features(x) # (batch_size, seq_length, n_levels, embedding_dim)

            # Step 2: Temporal encoding
            encoded_sequence, seq_length = self._encode_temporal_features(x_encoded) # (batch_size * seq_len, n_levels, embedding_dim)

            # Step 3 (Optional): Mask embeddings
            levels_mask = x["mask"].repeat_interleave(seq_length, dim=0)
            if self.mask_embeddings is not None:
                input_ids = x["input_ids"].repeat_interleave(seq_length, dim=0)
                encoded_sequence += self.mask_embeddings(input_ids)

            ### NEW LINE ###
            # Compute pooling of input to Transformer (skip connection path)
            pooled_input = self._pool_embeddings(encoded_sequence, levels_mask) # (batch_size * seq_len, embedding_dim)

            # Step 4: Transformer blocks
            transformed = self._apply_transformer_blocks(encoded_sequence, levels_mask) # (batch_size * seq_len, n_levels, embedding_dim)

            # Step 5: Pool embeddings
            ### MODIFIED LINE ###
            # Compute pooling of Transformer output
            pooled_transformed = self._pool_embeddings(transformed, levels_mask) # (batch_size * seq_len, embedding_dim)
            
            ### NEW LINE ###
            # Add skip connection (Input + Transformed) to create residual connection
            pooled_embedding = pooled_input + pooled_transformed # (batch_size * seq_len, embedding_dim)

            # Step 7: Decoder
            # Combine with parameters
            parameters = x["level_0"].contiguous().view(-1, self.parameters_dim) # (batch_size * seq_len, parameters_dim)
            combined_input = torch.cat((pooled_embedding, parameters), dim=1) # (batch_size * seq_len, embedding_dim + parameters_dim)
            
            #Reshape for GRU input
            decoder_input = combined_input.view(-1, seq_length, combined_input.size(-1)) # (batch_size, seq_length, embedding_dim + parameters_dim)
            
            # Pass through decoder
            gru_output, _ = self.decoder[0](decoder_input) # (batch_size, seq_length, embedding_dim)
            output = self.decoder[1](gru_output) # (batch_size, seq_length, output_dim)

            return output
    

    #####################################
    
        
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
        self.temporal_encoders = nn.ModuleList(
            [
                nn.GRU(input_size=dim, hidden_size=self.embedding_dim, batch_first=True)
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
            for level, encoder in enumerate(self.temporal_encoders, 1)
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
            combined = fc_hidden(combined) + combined

        out = self.fc_output(combined)
        
        return out.contiguous().view(-1, seq_length, self.output_dim)
    

#Alternative Decoders:

class CNN_Decoder(nn.Module):
    def __init__(self):
        # ... definisci i layer ConvTranspose2d ...
        self.mask_flat = torch.tensor(region_mask).flatten() # Maschera fissa salvata nel modello

    def forward(self, latent_vector):
        # 1. Reshape vettore -> immagine piccola
        x = latent_vector.view(-1, 16, 4, 4) 
        # 2. Upsampling
        x = self.conv_layers(x) # Diventa (Batch, 3_quantili, 216, 216)
        # 3. Masking finale
        x_flat = x.view(Batch, 3, -1) # Appiattisci spazialmente
        output = x_flat[..., self.mask_flat] # Tieni solo i pixel della regione
        return output

class INR_Decoder(nn.Module):
    def __init__(self, active_pixels_coords):
        super().__init__()
        # Buffer fisso con le coordinate (x,y) normalizzate dei 2477 pixel
        self.register_buffer('grid_coords', active_pixels_coords) 
        
        # MLP piccolo che lavora pixel per pixel
        self.mlp = nn.Sequential(
            nn.Linear(64 + 2, 128), # Latente + X + Y
            nn.ReLU(),
            nn.Linear(128, 3) # Output: 3 quantili per quel pixel
        )

    def forward(self, latent_z):
        # latent_z: (Batch, Seq, 64)
        # grid_coords: (N_pixels, 2)
        
        # Espandiamo latent_z per ogni pixel
        z_expanded = latent_z.unsqueeze(-2).expand(..., N_pixels, 64)
        
        # Espandiamo grid_coords per ogni batch/time
        coords_expanded = self.grid_coords.expand(Batch, Seq, N_pixels, 2)
        
        # Concateniamo
        mlp_input = torch.cat([z_expanded, coords_expanded], dim=-1)
        
        # Predizione
        return self.mlp(mlp_input)

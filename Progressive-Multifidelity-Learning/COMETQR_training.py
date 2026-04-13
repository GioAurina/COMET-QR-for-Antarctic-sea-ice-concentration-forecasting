"""Model setup and training components for the multifidelity pipeline."""

from COMETQR_shared import *

class ModelSetup:
    """Setup model, loss function, and optimizer"""
    
    def __init__(self, config, levels_dim, n_pixel_region, train_dataset):
        self.config = config
        self.levels_dim = levels_dim
        self.n_pixel_region = n_pixel_region
        self.train_dataset = train_dataset
        self.quantiles = [0.05, 0.5, 0.95]
        
    def create_model(self):
        """Create multifidelity transformer model"""
        print("\nCREATING MODEL")
        print("-"*70)
        
        output_dim = self.n_pixel_region * len(self.quantiles)
        n_masks = (self.train_dataset.mask_manager.n_masks 
                  if hasattr(self.train_dataset, 'mask_manager') else 1)
        
        model = MultifidelityTransformer(
            levels_dim=self.levels_dim,
            embedding_dim=int(self.config.parameters["model"]["embedding_dim"]),
            parameters_dim=1,  # x0 has dimension 1
            output_dim=output_dim,
            n_heads=int(self.config.parameters["model"]["n_heads"]),
            n_masks=(n_masks if self.config.parameters["model"]["mask_embeddings"] else None),
            n_transformer_blocks=int(self.config.parameters["model"]["n_transformer_blocks"]),
            spatial_encoders_dim={},
            dropout=0.2  # Increased from 0.1 to 0.2 to prevent overfitting
        ).to(self.config.device)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created with skip connections and 4×emb_dim FFN")
        print(f"  Parameters: {n_params:,}")
        print(f"  Output dimension: {output_dim}")
        print(f"  Quantiles: {self.quantiles}")
        print(f"  Architecture: {int(self.config.parameters['model']['n_transformer_blocks'])} Transformer blocks, emb_dim={int(self.config.parameters['model']['embedding_dim'])}, heads={int(self.config.parameters['model']['n_heads'])}")
        
        return model
        
    def create_loss_function(self):
        """Create full spatial quantile loss"""
        
        class FullSpatialQuantileLoss(nn.Module):
            def __init__(self, quantiles, n_pixels):
                super().__init__()
                self.quantiles = quantiles
                self.n_pixels = n_pixels
                
            def forward(self, preds, targets):
                B, S, _ = preds.shape
                preds_r = preds.view(B, S, len(self.quantiles), self.n_pixels)
                
                loss = 0.0
                for i, tau in enumerate(self.quantiles):
                    pred_q = preds_r[:, :, i, :]
                    err = targets - pred_q
                    # Compute quantile loss with in-place max to save memory
                    tau_weight = torch.where(err >= 0, tau, tau - 1)
                    loss += torch.mean(tau_weight * err)
                    # Free memory immediately
                    del pred_q, err, tau_weight
                    
                return loss
                
        loss_fn = FullSpatialQuantileLoss(self.quantiles, self.n_pixel_region)
        print(f"✓ Loss function created (Full region evaluation)")
        
        return loss_fn
        
    def create_optimizer_scheduler(self, model):
        """Create optimizer and learning rate scheduler"""
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config.parameters["training"]["optim_params"]["learning_rate"]),
            weight_decay=float(self.config.parameters["training"]["optim_params"]["weight_decay"])
        )
        
        lr_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: lr_schedule(
                step,
                int(self.config.parameters["model"]["embedding_dim"]),
                1.0,
                int(self.config.parameters["training"]["optim_params"]["warmup"])
            )
        )
        
        print(f"✓ Optimizer and scheduler created")
        
        return optimizer, lr_scheduler




class ModelTrainer:
    """Handle model training with checkpointing and early stopping"""
    
    def __init__(self, config, model, loss_fn, optimizer, lr_scheduler, 
                 train_loader, val_loader):
        self.config = config
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Paths
        self.checkpoint_path = config.checkpoint_dir / f"{config.experiment}.pt"
        self.latest_path = config.checkpoint_dir / f"{config.experiment}_latest.pt"
        self.history_path = config.checkpoint_dir / f"{config.experiment}_history.pkl"
        
        # Training state
        self.train_losses = []
        self.val_losses = []
        self.combinatorial_patterns = ["3L_all", "2L_1+2", "2L_1+3", "2L_2+3", "1L_1", "1L_2", "1L_3"]
        self.train_fidelity_losses = {p: [] for p in self.combinatorial_patterns}
        self.val_fidelity_losses = {p: [] for p in self.combinatorial_patterns}
        self.best_val_loss = float('inf')
        self.epochs_trained = 0
        
        # WandB
        self.wandb_run = None
        if config.parameters["logging"]["wandb"] and not config.args.no_wandb:
            wandb.login()
            self.wandb_run = wandb.init(
                project="MultifidelityTransformer",
                config=config.parameters,
                name=config.parameters["experiment_name"],
                reinit=True
            )
            
    def load_checkpoint(self):
        """Load model checkpoint if it exists"""
        if not self.config.parameters["model"]["load_model"]:
            return False
            
        print("\n" + "="*70)
        print("LOADING CHECKPOINT")
        print("="*70)
        
        if self.checkpoint_path.exists():
            state_dict = torch.load(self.checkpoint_path, map_location=self.config.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            print(f"✓ Model weights loaded from: {self.checkpoint_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
            
        if self.history_path.exists():
            with open(self.history_path, 'rb') as f:
                history = pickle.load(f)
                
            self.train_losses = history.get('train_loss', [])
            self.val_losses = history.get('val_loss', [])
            self.train_fidelity_losses = history.get('train_fidelity', {})
            self.val_fidelity_losses = history.get('val_fidelity', {})
            self.best_val_loss = history.get('best_val_loss', float('inf'))
            self.epochs_trained = history.get('epochs_trained', 0)
            
            print(f"✓ Training history loaded: {self.epochs_trained} epochs")
        else:
            print("⚠️  No training history found")
            
        return True
        
    def save_history(self):
        """Save training history"""
        history = {
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
            "train_fidelity": self.train_fidelity_losses,
            "val_fidelity": self.val_fidelity_losses,
            "best_val_loss": self.best_val_loss,
            "epochs_trained": self.epochs_trained,
            "parameters": self.config.parameters
        }
        
        with open(self.history_path, 'wb') as f:
            pickle.dump(history, f)
            
    def train(self):
        """Main training loop"""
        if self.load_checkpoint():
            print("\n✓ Model loaded. Skipping training.")
            return
            
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        
        epochs = int(self.config.parameters["training"]["epochs"])
        patience = 10
        patience_counter = 0
        best_model_state = None
        
        print(f"Epochs: {epochs}, Patience: {patience}")
        print("Press Ctrl+C to interrupt and save progress")
        
        # Watch model with WandB
        if self.wandb_run and self.config.parameters["logging"]["gradients"]:
            self.wandb_run.watch(self.model, log="all", log_freq=100)
            
        try:
            for epoch in range(epochs):
                # Train
                self.model.train()
                tr_res = run_epoch(
                    self.train_loader, self.model, self.loss_fn,
                    self.optimizer, self.lr_scheduler, 10,
                    epoch_n=epoch, wandb_run=self.wandb_run,
                    track_fidelity_losses=True
                )
                tr_loss = tr_res[0]
                tr_fid = tr_res[2] if len(tr_res) > 2 else {}
                
                self.train_losses.append(tr_loss)
                for p in self.combinatorial_patterns:
                    if tr_fid.get(p) is not None:
                        self.train_fidelity_losses[p].append(tr_fid[p])
                        
                # Validate
                self.model.eval()
                val_res = run_epoch(
                    self.val_loader, self.model, self.loss_fn,
                    self.optimizer, self.lr_scheduler, 10,
                    mode="eval", track_fidelity_losses=True
                )
                val_loss = val_res[0]
                val_fid = val_res[2] if len(val_res) > 2 else {}
                
                self.val_losses.append(val_loss)
                for p in self.combinatorial_patterns:
                    if val_fid.get(p) is not None:
                        self.val_fidelity_losses[p].append(val_fid[p])
                        
                # Logging (clean output, detailed tracking saved for plotting)
                print(f"Epoch {epoch+1}/{epochs} | Train: {tr_loss:.4f} | Val: {val_loss:.4f}")
                
                if self.wandb_run:
                    self.wandb_run.log({
                        "epoch": epoch,
                        "train_loss": tr_loss,
                        "val_loss": val_loss,
                        "best_val_loss": self.best_val_loss
                    })
                    
                # Save history (incremental)
                self.epochs_trained = epoch + 1
                self.save_history()
                
                # Early stopping - only save when finding new best
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model to disk immediately (no GPU memory retention)
                    torch.save(self.model.state_dict(), self.checkpoint_path)
                    print(f"  ✓ New best model saved (Val Loss: {val_loss:.4f})")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"\n🛑 Early stopping at epoch {epoch+1}")
                        break
                        
        except KeyboardInterrupt:
            print("\n🛑 Training interrupted by user")
            
        finally:
            # Restore best weights from checkpoint if it exists
            if self.checkpoint_path.exists():
                self.model.load_state_dict(
                    torch.load(self.checkpoint_path, map_location=self.config.device, weights_only=True)
                )
                print("✓ Best model weights restored from checkpoint")
                
            self.save_history()
            print(f"✓ Training state saved to: {self.history_path}")
            
        self.model.eval()
        print(f"\n✅ Training complete. Best Val Loss: {self.best_val_loss:.4f}")





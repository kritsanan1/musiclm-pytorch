#!/usr/bin/env python3
"""
Multi-GPU and distributed training enhancements for MusicLM PyTorch implementation.

This module provides:
- Distributed data parallel training support
- Multi-GPU training with efficient data loading
- Gradient synchronization and communication optimizations
- Mixed precision training for distributed setups
- Checkpoint management for distributed training
- Dynamic batching and load balancing
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Any, List, Tuple, Union
import os
import json
import logging
import time
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

from musiclm_pytorch import MuLaN, AudioLM, MusicLM
from musiclm_pytorch.trainer import MuLaNTrainer

logger = logging.getLogger(__name__)

@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = 'localhost'
    master_port: str = '12345'
    backend: str = 'nccl'
    use_ddp: bool = True
    use_amp: bool = True
    gradient_accumulation_steps: int = 1
    gradient_clipping: float = 1.0
    sync_batchnorm: bool = True
    find_unused_parameters: bool = False
    bucket_cap_mb: int = 25
    broadcast_buffers: bool = True
    
    # Performance optimizations
    use_gradient_checkpointing: bool = False
    use_memory_efficient_attention: bool = True
    compile_distributed: bool = False
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True


class DistributedSetup:
    """Handle distributed training setup and initialization."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.is_distributed = False
        self.device = None
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        
        self._setup_distributed()
    
    def _setup_distributed(self):
        """Initialize distributed training environment."""
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            # Environment variables set by torchrun or similar
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.rank = int(os.environ['RANK'])
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.is_distributed = self.world_size > 1
            
            if self.is_distributed:
                self._init_process_group()
        else:
            # Single GPU training
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            self.is_distributed = False
        
        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.local_rank}')
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device('cpu')
        
        logger.info(f"Distributed setup: world_size={self.world_size}, rank={self.rank}, "
                   f"local_rank={self.local_rank}, device={self.device}")
    
    def _init_process_group(self):
        """Initialize the distributed process group."""
        try:
            if 'MASTER_ADDR' not in os.environ:
                os.environ['MASTER_ADDR'] = self.config.master_addr
            if 'MASTER_PORT' not in os.environ:
                os.environ['MASTER_PORT'] = self.config.master_port
            
            dist.init_process_group(
                backend=self.config.backend,
                world_size=self.world_size,
                rank=self.rank
            )
            
            logger.info(f"Distributed process group initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize distributed process group: {e}")
            self.is_distributed = False
            self.world_size = 1
            self.rank = 0
    
    def cleanup(self):
        """Cleanup distributed training resources."""
        if self.is_distributed:
            dist.destroy_process_group()
            logger.info("Distributed process group destroyed")
    
    def barrier(self):
        """Synchronize all processes."""
        if self.is_distributed:
            dist.barrier()
    
    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce operation across all processes."""
        if self.is_distributed:
            dist.all_reduce(tensor)
        return tensor
    
    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast tensor from source rank to all ranks."""
        if self.is_distributed:
            dist.broadcast(tensor, src)
        return tensor
    
    def gather(self, tensor: torch.Tensor, dst: int = 0) -> Optional[List[torch.Tensor]]:
        """Gather tensors from all ranks to destination rank."""
        if self.is_distributed:
            if self.rank == dst:
                gathered_tensors = [torch.zeros_like(tensor) for _ in range(self.world_size)]
                dist.gather(tensor, gathered_tensors, dst)
                return gathered_tensors
            else:
                dist.gather(tensor, dst=dst)
        return None


class DistributedMuLaNTrainer(MuLaNTrainer):
    """Distributed training wrapper for MuLaN trainer."""
    
    def __init__(self, mulan: MuLaN, dataset, distributed_config: DistributedConfig, **kwargs):
        self.distributed_config = distributed_config
        self.distributed_setup = DistributedSetup(distributed_config)
        
        # Initialize base trainer with distributed-aware parameters
        super().__init__(
            mulan=mulan,
            dataset=dataset,
            use_mixed_precision=distributed_config.use_amp,
            use_gradient_checkpointing=distributed_config.use_gradient_checkpointing,
            **kwargs
        )
        
        # Setup distributed training
        self.scaler = GradScaler() if distributed_config.use_amp else None
        self._setup_distributed_model()
        self._setup_distributed_dataloader()
    
    def _setup_distributed_model(self):
        """Setup model for distributed training."""
        if self.distributed_setup.is_distributed and self.distributed_config.use_ddp:
            # Convert BatchNorm to SyncBatchNorm
            if self.distributed_config.sync_batchnorm:
                self.mulan = nn.SyncBatchNorm.convert_sync_batchnorm(self.mulan)
            
            # Wrap model with DDP
            self.mulan = DDP(
                self.mulan,
                device_ids=[self.distributed_setup.local_rank] if torch.cuda.is_available() else None,
                output_device=self.distributed_setup.local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=self.distributed_config.find_unused_parameters,
                bucket_cap_mb=self.distributed_config.bucket_cap_mb,
                broadcast_buffers=self.distributed_config.broadcast_buffers
            )
            
            logger.info("Model wrapped with DistributedDataParallel")
        
        # Move model to device
        self.mulan.to(self.distributed_setup.device)
    
    def _setup_distributed_dataloader(self):
        """Setup distributed data loader."""
        if self.distributed_setup.is_distributed:
            self.sampler = DistributedSampler(
                self.dataset,
                num_replicas=self.distributed_setup.world_size,
                rank=self.distributed_setup.rank,
                shuffle=True
            )
        else:
            self.sampler = None
    
    def train_step(self, text: torch.Tensor, audio: torch.Tensor,
                   text_mask: Optional[torch.Tensor] = None,
                   audio_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Distributed training step with gradient synchronization."""
        self.mulan.train()
        
        # Move data to device
        text = text.to(self.distributed_setup.device)
        audio = audio.to(self.distributed_setup.device)
        if text_mask is not None:
            text_mask = text_mask.to(self.distributed_setup.device)
        if audio_mask is not None:
            audio_mask = audio_mask.to(self.distributed_setup.device)
        
        # Forward pass with mixed precision
        if self.distributed_config.use_amp:
            with autocast():
                text_embed, audio_embed = self.mulan(text, audio, text_mask, audio_mask)
                loss = self._compute_loss(text_embed, audio_embed)
            
            # Scale loss and backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.distributed_config.gradient_clipping > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.mulan.parameters(),
                    self.distributed_config.gradient_clipping
                )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard training without mixed precision
            text_embed, audio_embed = self.mulan(text, audio, text_mask, audio_mask)
            loss = self._compute_loss(text_embed, audio_embed)
            loss.backward()
            
            # Gradient clipping
            if self.distributed_config.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.mulan.parameters(),
                    self.distributed_config.gradient_clipping
                )
            
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        
        # Gather loss from all processes
        if self.distributed_setup.is_distributed:
            loss_tensor = torch.tensor(loss.item()).to(self.distributed_setup.device)
            self.distributed_setup.all_reduce(loss_tensor)
            avg_loss = loss_tensor.item() / self.distributed_setup.world_size
        else:
            avg_loss = loss.item()
        
        return {
            'loss': avg_loss,
            'learning_rate': self.scheduler.get_last_lr()[0] if self.scheduler else 0.0,
            'rank': self.distributed_setup.rank
        }
    
    def _compute_loss(self, text_embed: torch.Tensor, audio_embed: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss between text and audio embeddings."""
        # Cosine similarity
        text_embed_norm = torch.nn.functional.normalize(text_embed, dim=-1)
        audio_embed_norm = torch.nn.functional.normalize(audio_embed, dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(text_embed_norm, audio_embed_norm.t()) * torch.exp(self.mulan.temperature)
        
        # Contrastive loss
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_text = torch.nn.functional.cross_entropy(logits, labels)
        loss_audio = torch.nn.functional.cross_entropy(logits.t(), labels)
        
        return (loss_text + loss_audio) / 2
    
    def save_checkpoint_distributed(self, path: str, epoch: int, step: int):
        """Save checkpoint in distributed training."""
        if self.distributed_setup.rank == 0:
            # Only rank 0 saves the checkpoint
            checkpoint = {
                'epoch': epoch,
                'step': step,
                'model_state_dict': self.mulan.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'distributed_config': self.distributed_config,
                'world_size': self.distributed_setup.world_size,
                'rank': self.distributed_setup.rank
            }
            
            torch.save(checkpoint, path)
            logger.info(f"Checkpoint saved: {path}")
        
        # Synchronize all processes
        self.distributed_setup.barrier()
    
    def load_checkpoint_distributed(self, path: str) -> Dict[str, Any]:
        """Load checkpoint in distributed training."""
        checkpoint = torch.load(path, map_location=self.distributed_setup.device)
        
        # Load state dict
        self.mulan.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded: {path}")
        
        return checkpoint


class DistributedDataLoader:
    """Enhanced data loader for distributed training."""
    
    def __init__(self, dataset, batch_size: int, distributed_config: DistributedConfig,
                 shuffle: bool = True, drop_last: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.distributed_config = distributed_config
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        self.setup = DistributedSetup(distributed_config)
        self.sampler = self._create_sampler()
        self.dataloader = self._create_dataloader()
    
    def _create_sampler(self):
        """Create distributed sampler."""
        if self.setup.is_distributed:
            return DistributedSampler(
                self.dataset,
                num_replicas=self.setup.world_size,
                rank=self.setup.rank,
                shuffle=self.shuffle,
                drop_last=self.drop_last
            )
        else:
            return None
    
    def _create_dataloader(self):
        """Create PyTorch data loader."""
        from torch.utils.data import DataLoader
        
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            shuffle=(self.sampler is None and self.shuffle),
            drop_last=self.drop_last,
            num_workers=self.distributed_config.num_workers,
            pin_memory=self.distributed_config.pin_memory,
            prefetch_factor=self.distributed_config.prefetch_factor,
            persistent_workers=self.distributed_config.persistent_workers
        )
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)
    
    def set_epoch(self, epoch: int):
        """Set epoch for distributed sampler."""
        if self.sampler:
            self.sampler.set_epoch(epoch)


class GradientSynchronization:
    """Handle gradient synchronization across multiple processes."""
    
    def __init__(self, distributed_setup: DistributedSetup):
        self.setup = distributed_setup
    
    def synchronize_gradients(self, model: nn.Module):
        """Synchronize gradients across all processes."""
        if not self.setup.is_distributed:
            return
        
        for param in model.parameters():
            if param.grad is not None:
                self.setup.all_reduce(param.grad.data)
    
    def synchronize_parameters(self, model: nn.Module):
        """Synchronize model parameters across all processes."""
        if not self.setup.is_distributed:
            return
        
        for param in model.parameters():
            self.setup.broadcast(param.data)


class DistributedCheckpointManager:
    """Handle checkpointing in distributed training."""
    
    def __init__(self, distributed_setup: DistributedSetup):
        self.setup = distributed_setup
    
    def save_distributed_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                                    epoch: int, step: int, path: str,
                                    metrics: Optional[Dict[str, float]] = None):
        """Save checkpoint in distributed training."""
        if self.setup.rank != 0:
            return
        
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'distributed_config': {
                'world_size': self.setup.world_size,
                'rank': self.setup.rank
            },
            'metrics': metrics or {}
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Distributed checkpoint saved: {path}")
    
    def load_distributed_checkpoint(self, path: str, model: nn.Module,
                                    optimizer: Optional[torch.optim.Optimizer] = None,
                                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Dict[str, Any]:
        """Load checkpoint in distributed training."""
        checkpoint = torch.load(path, map_location=self.setup.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Distributed checkpoint loaded: {path}")
        
        return checkpoint


class PerformanceMonitor:
    """Monitor performance metrics in distributed training."""
    
    def __init__(self, distributed_setup: DistributedSetup):
        self.setup = distributed_setup
        self.metrics = {}
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log performance metrics."""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append((step, value))
    
    def get_average_metrics(self, window_size: int = 100) -> Dict[str, float]:
        """Get average metrics over recent window."""
        averages = {}
        
        for key, values in self.metrics.items():
            if len(values) >= window_size:
                recent_values = [v for _, v in values[-window_size:]]
                averages[key] = np.mean(recent_values)
            elif values:
                recent_values = [v for _, v in values]
                averages[key] = np.mean(recent_values)
        
        return averages
    
    def save_metrics(self, path: str):
        """Save metrics to file."""
        if self.setup.rank == 0:
            with open(path, 'w') as f:
                json.dump(self.metrics, f, indent=2)


def create_distributed_config(**kwargs) -> DistributedConfig:
    """Create distributed configuration with sensible defaults."""
    return DistributedConfig(**kwargs)


def setup_distributed_training(
    world_size: int = 1,
    rank: int = 0,
    local_rank: int = 0,
    master_addr: str = 'localhost',
    master_port: str = '12345',
    backend: str = 'nccl'
) -> DistributedSetup:
    """Setup distributed training environment."""
    config = DistributedConfig(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        master_addr=master_addr,
        master_port=master_port,
        backend=backend
    )
    
    return DistributedSetup(config)


def run_distributed_training(
    model: MuLaN,
    dataset,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    world_size: int = 1,
    **kwargs
) -> Dict[str, Any]:
    """Run distributed training with automatic setup."""
    
    # Setup distributed environment
    distributed_setup = setup_distributed_training(world_size=world_size)
    
    # Create distributed config
    distributed_config = DistributedConfig(
        world_size=world_size,
        rank=distributed_setup.rank,
        local_rank=distributed_setup.local_rank,
        use_ddp=True,
        use_amp=True,
        **{k: v for k, v in kwargs.items() if k in DistributedConfig.__dataclass_fields__}
    )
    
    # Create distributed trainer
    trainer = DistributedMuLaNTrainer(
        mulan=model,
        dataset=dataset,
        distributed_config=distributed_config,
        lr=learning_rate,
        **{k: v for k, v in kwargs.items() if k not in DistributedConfig.__dataclass_fields__}
    )
    
    # Create distributed data loader
    dataloader = DistributedDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        distributed_config=distributed_config,
        shuffle=True
    )
    
    # Training loop
    results = {
        'losses': [],
        'learning_rates': [],
        'epoch_times': [],
        'distributed_metrics': {}
    }
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        dataloader.set_epoch(epoch)
        
        epoch_losses = []
        epoch_lr = []
        
        for batch_idx, (text, audio, text_mask, audio_mask) in enumerate(dataloader):
            step_result = trainer.train_step(text, audio, text_mask, audio_mask)
            
            epoch_losses.append(step_result['loss'])
            epoch_lr.append(step_result['learning_rate'])
            
            if batch_idx % 100 == 0 and distributed_setup.rank == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {step_result['loss']:.4f}")
        
        epoch_time = time.time() - epoch_start
        avg_loss = np.mean(epoch_losses)
        avg_lr = np.mean(epoch_lr)
        
        results['losses'].append(avg_loss)
        results['learning_rates'].append(avg_lr)
        results['epoch_times'].append(epoch_time)
        
        if distributed_setup.rank == 0:
            logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = f"checkpoint_epoch_{epoch}.pt"
        trainer.save_checkpoint_distributed(checkpoint_path, epoch, batch_idx)
    
    # Cleanup
    distributed_setup.cleanup()
    
    return results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample model and dataset
    from musiclm_pytorch import MuLaN
    
    # This would be replaced with actual model and dataset
    model = MuLaN(
        audio_transformer_dim=512,
        text_transformer_dim=512,
        dim=512
    )
    
    # Create dummy dataset for demonstration
    class DummyDataset:
        def __init__(self, size=1000):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return (
                torch.randint(0, 1000, (77,)),  # text
                torch.randn(80, 1024),  # audio
                torch.ones(77),  # text_mask
                torch.ones(1024)  # audio_mask
            )
    
    dataset = DummyDataset(1000)
    
    # Run distributed training
    results = run_distributed_training(
        model=model,
        dataset=dataset,
        num_epochs=2,
        batch_size=16,
        learning_rate=1e-4,
        world_size=1  # Change to >1 for actual distributed training
    )
    
    logger.info("Distributed training completed!")
    logger.info(f"Final results: {results}")
    
    # Test distributed setup
    config = DistributedConfig(world_size=1, rank=0, local_rank=0)
    setup = DistributedSetup(config)
    logger.info(f"Distributed setup test: rank={setup.rank}, world_size={setup.world_size}")
    setup.cleanup()
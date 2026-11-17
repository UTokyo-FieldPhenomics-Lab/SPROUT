"""Train SPROUT from unlabeled RGB images (PyTorch Lightning 1.4.2)."""

import argparse
import importlib
import os
from pathlib import Path

from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import torch
from torch.utils.data import DataLoader

from sprout.pretraining import DiffusionPretraining, PretrainingImages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/sprout-S-pretrain.yaml')
    parser.add_argument('--image-list', required=True, help='One image path per line.')
    parser.add_argument('--data-root', default='.', help='Root for paths in the image list.')
    parser.add_argument('--output-dir', required=True, help='A new run directory.')
    parser.add_argument('--resume', help='Full checkpoint from this pretraining entry point.')
    parser.add_argument('--gpus', type=int, default=1, help='GPU count; 0 for CPU.')
    parser.add_argument('--precision', type=int, choices=[16, 32], default=32)
    parser.add_argument('--max-steps', type=int, help='Override optimizer step count.')
    parser.add_argument('--batch-size', type=int, help='Override batch size per GPU.')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--accumulate-grad-batches', type=int, default=1)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    training = config.training
    if args.max_steps is not None:
        training.max_steps = args.max_steps
    if args.batch_size is not None:
        training.batch_size = args.batch_size
    if min(training.max_steps, training.batch_size, training.save_every,
           args.accumulate_grad_batches) <= 0 or min(args.num_workers, args.gpus) < 0:
        parser.error('Steps, batch size, save interval and accumulation must be positive; workers/GPUs nonnegative.')
    if args.precision == 16 and args.gpus == 0:
        parser.error('FP16 training requires a GPU; use --precision 32 for CPU.')
    if args.gpus > torch.cuda.device_count():
        parser.error('Requested GPU count exceeds the available CUDA devices.')
    resume = Path(args.resume).resolve() if args.resume else None
    if resume is not None and not resume.is_file():
        parser.error(f'Checkpoint does not exist: {resume}')
    output = Path(args.output_dir).resolve()
    if int(os.environ.get('LOCAL_RANK', 0)) == 0 and output.exists() and any(output.iterdir()):
        if resume is None or resume.parent != output:
            parser.error('Output directory is not empty. Use a fresh directory, or resume a checkpoint in this run.')
        saved = OmegaConf.load(output / 'pretraining.yaml')
        if OmegaConf.to_container(saved.model) != OmegaConf.to_container(config.model):
            parser.error('Resume model config differs from the saved run.')
        if OmegaConf.to_container(saved.diffusion) != OmegaConf.to_container(config.diffusion):
            parser.error('Resume diffusion config differs from the saved run.')
        if saved.training.learning_rate != training.learning_rate or saved.training.warmup_steps != training.warmup_steps:
            parser.error('Resume learning rate and warmup must match the saved run.')
    pl.seed_everything(training.seed, workers=True)
    dataset = PretrainingImages(args.image_list, args.data_root,
                                size=config.model.params.image_size)
    loader = DataLoader(dataset, batch_size=training.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=args.gpus > 0)
    module_name, class_name = config.model.target.rsplit('.', 1)
    backbone = getattr(importlib.import_module(module_name), class_name)(**config.model.params)
    model = DiffusionPretraining(backbone, learning_rate=training.learning_rate,
                                warmup_steps=training.warmup_steps, **config.diffusion)
    output.mkdir(parents=True, exist_ok=True)
    # Spawned DDP workers re-enter this script; only the original process writes metadata.
    if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        OmegaConf.save(config, output / 'pretraining.yaml')
        OmegaConf.save(OmegaConf.create(vars(args)), output / 'arguments.yaml')
    checkpoint = ModelCheckpoint(dirpath=str(output), filename='SPROUT-{step}',
                                 every_n_train_steps=training.save_every,
                                 save_top_k=-1, save_last=True, save_weights_only=False)
    trainer = pl.Trainer(gpus=args.gpus, accelerator='ddp' if args.gpus > 1 else None,
                         precision=args.precision, max_steps=training.max_steps,
                         max_epochs=None, callbacks=[checkpoint],
                         logger=CSVLogger(str(output), name='metrics'),
                         accumulate_grad_batches=args.accumulate_grad_batches,
                         resume_from_checkpoint=str(resume) if resume else None,
                         num_sanity_val_steps=0, log_every_n_steps=10)
    trainer.fit(model, train_dataloaders=loader)
    # Also save when max_steps is not a multiple of save_every.
    trainer.save_checkpoint(str(output / 'last.ckpt'))


if __name__ == '__main__':
    main()

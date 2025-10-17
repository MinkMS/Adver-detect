#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EfficientNet_b0_train.py

Train EfficientNet-B0 on Food-101.
 - Supports: train-from-scratch (--from_scratch) or pretrained (default)
 - Augmentations: same as resnet script (RandomResizedCrop, ColorJitter, RandomRotation)
 - Features: AMP, gradient accumulation, freeze_head for first N epochs, Cosine lr, AdamW,
             CSV + TensorBoard logging, early stopping, best/final checkpoints.
 - Config: edit the CONFIG block at the top or use CLI args to override.
Style: spacing-heavy (operators, commas, args spaced) as requested.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import random
import time
from pathlib import Path
from typing import Dict , Tuple , Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets , transforms

try:
    import timm
    _HAVE_TIMM = True
except Exception:
    _HAVE_TIMM = False

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAVE_TB = True
except Exception:
    _HAVE_TB = False

try:
    from tqdm import tqdm
    _HAVE_TQDM = True
except Exception:
    _HAVE_TQDM = False

# -------------------------
# CONFIG block (edit here)
# -------------------------
# Using the same default input path as resnet18-v3.py (ref)
DEFAULT_INPUT_DIR       = r"C:\Users\Mink\OneDrive\Documents\Coding\Datasets\Food-101\food-101-split"
DEFAULT_OUTPUT_DIR      = r"C:\Users\Mink\OneDrive\Documents\Coding\GitHub\Adver-detect\effnet_b0"
DEFAULT_BATCH_SIZE      = 64
DEFAULT_NUM_WORKERS     = 8
DEFAULT_EPOCHS          = 30
DEFAULT_LR              = 1e-4
DEFAULT_WEIGHT_DECAY    = 1e-4
DEFAULT_SEED            = 4
EARLY_STOP_PATIENCE     = 5
# -------------------------

def set_seed( seed : int = 4 ) -> None:
    random.seed( seed )
    np.random.seed( seed )
    torch.manual_seed( seed )
    torch.cuda.manual_seed_all( seed )
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def get_data_loaders( data_dir : str , batch_size : int , num_workers : int ) -> Tuple[ Dict[str , DataLoader ] , Dict[str , int ] , list ]:
    """
    Data loaders: train/val/test using ImageFolder.
    Augmentations same as ResNet script.
    """
    train_transforms = transforms.Compose( [
        transforms.RandomResizedCrop( 224 , scale = ( 0.7 , 1.0 ) ),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter( brightness = 0.2 , contrast = 0.2 , saturation = 0.2 , hue = 0.1 ),
        transforms.RandomRotation( 15 ),
        transforms.ToTensor(),
        transforms.Normalize( [ 0.485 , 0.456 , 0.406 ] , [ 0.229 , 0.224 , 0.225 ] ),
    ] )

    eval_transforms = transforms.Compose( [
        transforms.Resize( 256 ),
        transforms.CenterCrop( 224 ),
        transforms.ToTensor(),
        transforms.Normalize( [ 0.485 , 0.456 , 0.406 ] , [ 0.229 , 0.224 , 0.225 ] ),
    ] )

    image_datasets = {
        'train' : datasets.ImageFolder( os.path.join( data_dir , 'train' ) , train_transforms ),
        'val'   : datasets.ImageFolder( os.path.join( data_dir , 'val' ) , eval_transforms ),
        'test'  : datasets.ImageFolder( os.path.join( data_dir , 'test' ) , eval_transforms ),
    }

    dataloaders = {
        x : DataLoader(
            image_datasets[ x ] ,
            batch_size = batch_size ,
            shuffle = ( x == 'train' ) ,
            num_workers = num_workers ,
            pin_memory = True
        )
        for x in [ 'train' , 'val' , 'test' ]
    }

    dataset_sizes = { x : len( image_datasets[ x ] ) for x in [ 'train' , 'val' , 'test' ] }
    class_names = image_datasets[ 'train' ].classes
    return dataloaders , dataset_sizes , class_names

def build_effnet_b0( num_classes : int , pretrained : bool = True , drop_rate : float = 0.2 ) -> nn.Module:
    """
    Build EfficientNet-b0 using timm. If pretrained False, trains from scratch.
    Assumes timm installed. Replace classifier head according to num_classes.
    """
    if not _HAVE_TIMM:
        raise RuntimeError( "timm is required for EfficientNet. pip install timm" )
    # create model with no classifier then attach our head
    model = timm.create_model( 'efficientnet_b0' , pretrained = pretrained , num_classes = 0 , global_pool = 'avg' )
    feat_dim = model.num_features
    # simple classifier head with dropout
    head = nn.Sequential(
        nn.Dropout( p = drop_rate ),
        nn.Linear( feat_dim , num_classes )
    )
    model.classifier = head
    return model

def freeze_backbone_timm( model : nn.Module ) -> None:
    """
    Freeze all timm backbone parameters except classifier head.
    """
    for name , param in model.named_parameters():
        if 'classifier' in name or 'head' in name or name.startswith( 'classifier' ):
            param.requires_grad = True
        else:
            param.requires_grad = False

def unfreeze_backbone( model : nn.Module ) -> None:
    for param in model.parameters():
        param.requires_grad = True

def evaluate( model : nn.Module , dataloader : DataLoader , device : torch.device , use_amp : bool = False ) -> Tuple[ float , float ]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for inputs , labels in dataloader:
            inputs = inputs.to( device , non_blocking = True )
            labels = labels.to( device , non_blocking = True )
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model( inputs )
                    loss = criterion( outputs , labels )
            else:
                outputs = model( inputs )
                loss = criterion( outputs , labels )

            _, preds = torch.max( outputs , 1 )
            bs = inputs.size( 0 )
            running_loss += loss.item() * bs
            running_corrects += torch.sum( preds == labels ).item()
            total += bs

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = float( running_corrects ) / float( total ) if total > 0 else 0.0
    return epoch_loss , epoch_acc

def train_one_epoch( model : nn.Module , dataloader : DataLoader , optimizer : optim.Optimizer , device : torch.device , criterion , use_amp : bool = False , accum_steps : int = 1 ) -> Tuple[ float , float ]:
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    iterator = dataloader
    if _HAVE_TQDM:
        iterator = tqdm( dataloader , desc = "train batch" )

    optimizer.zero_grad()
    step = 0
    for inputs , labels in iterator:
        inputs = inputs.to( device , non_blocking = True )
        labels = labels.to( device , non_blocking = True )

        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model( inputs )
                loss = criterion( outputs , labels ) / float( accum_steps )
            scaler.scale( loss ).backward()
            if ( step + 1 ) % accum_steps == 0:
                scaler.step( optimizer )
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model( inputs )
            loss = criterion( outputs , labels ) / float( accum_steps )
            loss.backward()
            if ( step + 1 ) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        _, preds = torch.max( outputs , 1 )
        bs = inputs.size( 0 )
        running_loss += loss.item() * accum_steps * bs   # revert division when accumulating
        running_corrects += torch.sum( preds == labels ).item()
        total_samples += bs
        step += 1

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    epoch_acc = float( running_corrects ) / float( total_samples ) if total_samples > 0 else 0.0
    return epoch_loss , epoch_acc

def train( dataloaders : Dict[str , DataLoader ] , dataset_sizes : Dict[str , int ] , model : nn.Module , device : torch.device , output_dir : Path , num_epochs : int = 30 , lr : float = 1e-4 , weight_decay : float = 1e-4 , patience : int = 5 , seed : int = 4 , use_amp : bool = False , freeze_epochs : int = 0 , accum_steps : int = 1 ) -> None:
    set_seed( seed )
    model = model.to( device )

    criterion = nn.CrossEntropyLoss( label_smoothing = 0.1 )
    # optimizer uses only parameters that require grad (works if some frozen)
    optimizer = optim.AdamW( filter( lambda p : p.requires_grad , model.parameters() ) , lr = lr , weight_decay = weight_decay )
    scheduler = optim.lr_scheduler.CosineAnnealingLR( optimizer , T_max = num_epochs )

    output_dir.mkdir( parents = True , exist_ok = True )
    csv_path = output_dir / 'training_log.csv'
    tb_writer = None
    if _HAVE_TB:
        try:
            tb_writer = SummaryWriter( log_dir = str( output_dir / 'tensorboard' ) )
        except Exception:
            tb_writer = None

    csv_fields = [ 'epoch' , 'train_loss' , 'train_acc' , 'val_loss' , 'val_acc' , 'lr' , 'epoch_time_sec' ]
    with csv_path.open( 'w' , newline = '' , encoding = 'utf-8' ) as cf:
        writer = csv.DictWriter( cf , fieldnames = csv_fields )
        writer.writeheader()

    best_model_wts = copy.deepcopy( model.state_dict() )
    best_val_acc = 0.0
    no_improve = 0

    # initial freeze if requested (freeze classifier only train)
    if freeze_epochs > 0:
        freeze_backbone_timm( model )
        print( f"Backbone frozen for first { freeze_epochs } epochs (only classifier trained)." )

    for epoch in range( num_epochs ):
        epoch_start = time.time()

        # unfreeze logic after freeze_epochs
        if epoch == freeze_epochs:
            print( f"Unfreezing backbone at epoch { epoch + 1 }." )
            unfreeze_backbone( model )
            optimizer = optim.AdamW( filter( lambda p : p.requires_grad , model.parameters() ) , lr = lr , weight_decay = weight_decay )
            scheduler = optim.lr_scheduler.CosineAnnealingLR( optimizer , T_max = max( 1 , num_epochs - epoch ) )

        train_loss , train_acc = train_one_epoch( model , dataloaders[ 'train' ] , optimizer , device , criterion , use_amp = use_amp , accum_steps = accum_steps )
        val_loss , val_acc = evaluate( model , dataloaders[ 'val' ] , device , use_amp = use_amp )

        scheduler.step()
        current_lr = optimizer.param_groups[ 0 ][ 'lr' ]
        epoch_time = time.time() - epoch_start

        # Logging CSV
        with csv_path.open( 'a' , newline = '' , encoding = 'utf-8' ) as cf:
            writer = csv.DictWriter( cf , fieldnames = csv_fields )
            writer.writerow( {
                'epoch' : epoch + 1 ,
                'train_loss' : train_loss ,
                'train_acc' : train_acc ,
                'val_loss' : val_loss ,
                'val_acc' : val_acc ,
                'lr' : current_lr ,
                'epoch_time_sec' : int( epoch_time )
            } )

        if tb_writer is not None:
            tb_writer.add_scalar( 'train/loss' , train_loss , epoch + 1 )
            tb_writer.add_scalar( 'train/acc' , train_acc , epoch + 1 )
            tb_writer.add_scalar( 'val/loss' , val_loss , epoch + 1 )
            tb_writer.add_scalar( 'val/acc' , val_acc , epoch + 1 )
            tb_writer.add_scalar( 'lr' , current_lr , epoch + 1 )

        print( f"Epoch { epoch + 1 } / { num_epochs }  - train_loss: { train_loss :.4f}  train_acc: { train_acc :.4f}  val_loss: { val_loss :.4f}  val_acc: { val_acc :.4f}  lr: { current_lr :.6e}  time: { int( epoch_time ) }s" )

        # Early stopping & checkpoint best
        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy( model.state_dict() )
            best_path = output_dir / 'best_model.pth'
            torch.save( { 'model_state_dict' : best_model_wts , 'epoch' : epoch + 1 , 'val_acc' : best_val_acc } , best_path )
            print( f"  Saved best model to { best_path } ( val_acc = { best_val_acc :.4f} )" )
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print( f"  No improvement for { patience } epochs. Early stopping." )
                break

    # end epochs
    model.load_state_dict( best_model_wts )
    final_path = output_dir / 'final_model.pth'
    torch.save( { 'model_state_dict' : model.state_dict() } , final_path )
    print( f"Training finished. Best val_acc = { best_val_acc :.4f} . Final models saved to { final_path }" )

    if tb_writer is not None:
        tb_writer.close()

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser( description = "Train EfficientNet-B0 on Food-101 (from scratch or pretrained). Supports AMP, freeze epochs, accum steps." )
    p.add_argument( "--input" , "-i" , default = DEFAULT_INPUT_DIR , help = "Path to dataset root (contains train/ , val/ , test/)" )
    p.add_argument( "--output" , "-o" , default = DEFAULT_OUTPUT_DIR , help = "Output folder for logs and checkpoints" )
    p.add_argument( "--batch_size" , "-b" , type = int , default = DEFAULT_BATCH_SIZE , help = "Batch size" )
    p.add_argument( "--num_workers" , type = int , default = DEFAULT_NUM_WORKERS , help = "Number of data loader workers" )
    p.add_argument( "--epochs" , type = int , default = DEFAULT_EPOCHS , help = "Number of epochs" )
    p.add_argument( "--lr" , type = float , default = DEFAULT_LR , help = "Initial learning rate" )
    p.add_argument( "--weight_decay" , type = float , default = DEFAULT_WEIGHT_DECAY , help = "Weight decay for optimizer" )
    p.add_argument( "--seed" , type = int , default = DEFAULT_SEED , help = "Random seed" )
    p.add_argument( "--patience" , type = int , default = EARLY_STOP_PATIENCE , help = "Early stopping patience" )
    p.add_argument( "--use_amp" , action = "store_true" , help = "Use mixed precision (AMP) if CUDA available" )
    p.add_argument( "--freeze_epochs" , type = int , default = 0 , help = "Freeze backbone for first N epochs (only classifier trained)" )
    p.add_argument( "--accum_steps" , type = int , default = 1 , help = "Gradient accumulation steps to simulate larger batch" )
    p.add_argument( "--from_scratch" , action = "store_true" , help = "Train EfficientNet-B0 from scratch (no pretrained weights)" )
    return p.parse_args()

def main():
    args = parse_args()

    print( "Configuration:" )
    print( f"  input: { args.input }" )
    print( f"  output: { args.output }" )
    print( f"  batch_size: { args.batch_size } , num_workers: { args.num_workers } , epochs: { args.epochs }" )
    print( f"  lr: { args.lr } , weight_decay: { args.weight_decay } , seed: { args.seed }" )
    print( f"  use_amp: { args.use_amp } , freeze_epochs: { args.freeze_epochs } , accum_steps: { args.accum_steps } , from_scratch: { args.from_scratch }" )

    set_seed( args.seed )

    dataloaders , dataset_sizes , class_names = get_data_loaders( args.input , args.batch_size , args.num_workers )
    num_classes = len( class_names )
    print( f"Detected { num_classes } classes, dataset sizes: { dataset_sizes }" )

    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
    print( f"Using device: { device }" )

    model = build_effnet_b0( num_classes = num_classes , pretrained = ( not args.from_scratch ) , drop_rate = 0.2 )

    train( dataloaders , dataset_sizes , model , device , Path( args.output ) , num_epochs = args.epochs , lr = args.lr , weight_decay = args.weight_decay , patience = args.patience , seed = args.seed , use_amp = args.use_amp , freeze_epochs = args.freeze_epochs , accum_steps = args.accum_steps )

if __name__ == "__main__":
    main()

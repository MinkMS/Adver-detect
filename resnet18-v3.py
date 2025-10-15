#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resnet18_on_Food101_train_v2_plus.py

Phiên bản mở rộng của Resnet18_on_Food101_train.py
 - Thêm Mixed Precision (AMP) với --use_amp
 - Thêm Gradual Unfreezing (freeze backbone N epochs) với --freeze_epochs
 - Thêm optional Optuna hyperparameter sweep với --optuna_trials
 - Vẫn giữ: pretrained resnet18, augmentation, AdamW, CosineAnnealingLR, label smoothing, dropout, early stopping, CSV + TensorBoard logging

Usage examples:
  python Resnet18_on_Food101_train_v2_plus.py --input ./food101_split --output ./runs/resnet18_v2 --epochs 30 --use_amp --freeze_epochs 5
  python Resnet18_on_Food101_train_v2_plus.py --optuna_trials 20 --input ./food101_split

Notes:
  - Optuna: must install `optuna` to use sweep.
  - AMP works only when CUDA available.
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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets , models , transforms

# Optional: TensorBoard, tqdm, optuna
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

try:
    import optuna
    _HAVE_OPTUNA = True
except Exception:
    _HAVE_OPTUNA = False

import pandas as pd

# -------------------------
# CONFIG block (edit here)
# -------------------------
DEFAULT_INPUT_DIR       = r"C:\Users\Mink\OneDrive\Documents\Coding\Datasets\Food-101\food-101-split"
DEFAULT_OUTPUT_DIR      = r"C:\Users\Mink\OneDrive\Documents\Coding\GitHub\Adver-detect\resnet18-ver3"
DEFAULT_BATCH_SIZE      = 64
DEFAULT_NUM_WORKERS     = 12
DEFAULT_EPOCHS          = 30
DEFAULT_LR              = 1e-3
DEFAULT_WEIGHT_DECAY    = 5e-4
DEFAULT_SEED            = 4
EARLY_STOP_PATIENCE     = 5
# -------------------------

def set_seed( seed : int = 4 ) -> None:
    random.seed( seed )
    torch.manual_seed( seed )
    torch.cuda.manual_seed_all( seed )
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def get_data_loaders( data_dir : str , batch_size : int , num_workers : int ) -> Tuple[ Dict[str , DataLoader ] , Dict[str , int ] , list ]:
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

def build_resnet18( num_classes : int , dropout_p : float = 0.5 ) -> nn.Module:
    model = models.resnet18( weights = models.ResNet18_Weights.IMAGENET1K_V1 )
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout( p = dropout_p ),
        nn.Linear( in_features , num_classes )
    )
    return model

def freeze_backbone( model : nn.Module ) -> None:
    """
    Freeze all parameters except the final classifier (fc).
    """
    for name , param in model.named_parameters():
        if not name.startswith( 'fc' ):
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
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

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

def train_one_epoch( model : nn.Module , dataloader : DataLoader , optimizer : optim.Optimizer , device : torch.device , criterion , use_amp : bool = False ) -> Tuple[ float , float ]:
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    iterator = dataloader
    if _HAVE_TQDM:
        iterator = tqdm( dataloader , desc = "train batch" )

    for inputs , labels in iterator:
        inputs = inputs.to( device , non_blocking = True )
        labels = labels.to( device , non_blocking = True )

        optimizer.zero_grad()
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model( inputs )
                loss = criterion( outputs , labels )
            scaler.scale( loss ).backward()
            scaler.step( optimizer )
            scaler.update()
        else:
            outputs = model( inputs )
            loss = criterion( outputs , labels )
            loss.backward()
            optimizer.step()

        bs = inputs.size( 0 )
        running_loss += loss.item() * bs
        _, preds = torch.max( outputs , 1 )
        running_corrects += torch.sum( preds == labels ).item()
        total_samples += bs

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    epoch_acc = float( running_corrects ) / float( total_samples ) if total_samples > 0 else 0.0
    return epoch_loss , epoch_acc

def train( dataloaders : Dict[str , DataLoader ] , dataset_sizes : Dict[str , int ] , model : nn.Module , device : torch.device , output_dir : Path , num_epochs : int = 30 , lr : float = 1e-3 , weight_decay : float = 5e-4 , patience : int = 5 , seed : int = 4 , use_amp : bool = False , freeze_epochs : int = 0 ) -> None:
    set_seed( seed )
    model = model.to( device )

    criterion = nn.CrossEntropyLoss( label_smoothing = 0.1 )
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

    # initial freeze if requested
    if freeze_epochs > 0:
        freeze_backbone( model )
        print( f"Backbone frozen for first { freeze_epochs } epochs (only fc trained)." )

    for epoch in range( num_epochs ):
        epoch_start = time.time()

        # unfreeze logic: after freeze_epochs completed, unfreeze once
        if epoch == freeze_epochs:
            print( f"Unfreezing backbone at epoch { epoch + 1 }." )
            unfreeze_backbone( model )
            # recreate optimizer to include newly unfrozen params
            optimizer = optim.AdamW( filter( lambda p : p.requires_grad , model.parameters() ) , lr = lr , weight_decay = weight_decay )
            scheduler = optim.lr_scheduler.CosineAnnealingLR( optimizer , T_max = num_epochs - epoch )

        train_loss , train_acc = train_one_epoch( model , dataloaders[ 'train' ] , optimizer , device , criterion , use_amp = use_amp )
        val_loss , val_acc = evaluate( model , dataloaders[ 'val' ] , device , use_amp = use_amp )

        # step scheduler
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

        print( f"Epoch { epoch + 1 } / { num_epochs }  - train_loss: {train_loss :.4f}  train_acc: {train_acc :.4f}  val_loss: {val_loss :.4f}  val_acc: {val_acc :.4f}  lr: {current_lr :.6e}  time: { int( epoch_time ) }s" )

        # Early stopping & checkpoint best
        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy( model.state_dict() )
            best_path = output_dir / 'best_model.pth'
            torch.save( { 'model_state_dict' : best_model_wts , 'epoch' : epoch + 1 , 'val_acc' : best_val_acc } , best_path )
            print( f"  Saved best model to { best_path } ( val_acc = { best_val_acc :.4f } )" )
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
    print( f"Training finished. Best val_acc = { best_val_acc :.4f } . Final models saved to { final_path }" )

    if tb_writer is not None:
        tb_writer.close()

# -------------------------
# Optuna support
# -------------------------
def objective_optuna( trial : 'optuna.Trial' , args ) -> float:
    """
    Optuna objective function.
    Sample hyperparameters, run short training, return validation accuracy.
    """
    # sample
    lr = trial.suggest_loguniform( 'lr' , 1e-5 , 1e-2 )
    weight_decay = trial.suggest_loguniform( 'weight_decay' , 1e-6 , 1e-2 )
    dropout_p = trial.suggest_uniform( 'dropout_p' , 0.2 , 0.7 )
    batch_size = trial.suggest_categorical( 'batch_size' , [ 32 , 64 ] )

    # build data loaders with sampled batch size
    dataloaders , dataset_sizes , class_names = get_data_loaders( args.input , batch_size , args.num_workers )

    num_classes = len( class_names )
    model = build_resnet18( num_classes = num_classes , dropout_p = dropout_p )

    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
    # For quick optuna trials, run fewer epochs (e.g., 5)
    n_epochs = min( 8 , args.epochs )
    train( dataloaders , dataset_sizes , model , device , Path( args.output ) / f'optuna_trial_{ trial.number }' , num_epochs = n_epochs , lr = lr , weight_decay = weight_decay , patience = 3 , seed = args.seed , use_amp = args.use_amp , freeze_epochs = args.freeze_epochs )
    # read val acc from the trial output csv
    csv_path = Path( args.output ) / f'optuna_trial_{ trial.number }' / 'training_log.csv'
    if not csv_path.exists():
        return 0.0
    df = pd.read_csv( csv_path )
    if 'val_acc' in df.columns:
        val_acc = float( df[ 'val_acc' ].max() )
    else:
        val_acc = 0.0
    return val_acc

def run_optuna( args ) -> None:
    if not _HAVE_OPTUNA:
        print( "Optuna not installed. Install optuna to use hyperparameter sweep ( pip install optuna )." )
        return
    print( f"Running Optuna with { args.optuna_trials } trials ..." )
    study = optuna.create_study( direction = 'maximize' )
    func = lambda trial : objective_optuna( trial , args )
    study.optimize( func , n_trials = args.optuna_trials )
    print( "Optuna best trial:" )
    print( study.best_trial.params )
    # Save best params
    best_path = Path( args.output ) / 'optuna_best_params.yaml'
    import yaml
    with best_path.open( 'w' , encoding = 'utf-8' ) as f:
        yaml.dump( study.best_trial.params , f )
    print( f"Saved best params to { best_path }" )

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser( description = "Fine-tune ResNet18 on Food-101 (AMP + Gradual Unfreeze + Optuna support)" )
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
    p.add_argument( "--freeze_epochs" , type = int , default = 0 , help = "Freeze backbone for first N epochs (only fc trained)" )
    p.add_argument( "--optuna_trials" , type = int , default = 0 , help = "If >0 and optuna installed, run hyperparameter sweep with given trials" )
    return p.parse_args()

def main():
    args = parse_args()

    print( "Configuration:" )
    print( f"  input: { args.input }" )
    print( f"  output: { args.output }" )
    print( f"  batch_size: { args.batch_size } , num_workers: { args.num_workers } , epochs: { args.epochs }" )
    print( f"  lr: { args.lr } , weight_decay: { args.weight_decay } , seed: { args.seed }" )
    print( f"  use_amp: { args.use_amp } , freeze_epochs: { args.freeze_epochs } , optuna_trials: { args.optuna_trials }" )

    if args.optuna_trials > 0:
        run_optuna( args )
        return

    set_seed( args.seed )

    dataloaders , dataset_sizes , class_names = get_data_loaders( args.input , args.batch_size , args.num_workers )
    num_classes = len( class_names )
    print( f"Detected { num_classes } classes, dataset sizes: { dataset_sizes }" )

    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
    print( f"Using device: { device }" )

    model = build_resnet18( num_classes = num_classes , dropout_p = 0.5 )

    train( dataloaders , dataset_sizes , model , device , Path( args.output ) , num_epochs = args.epochs , lr = args.lr , weight_decay = args.weight_decay , patience = args.patience , seed = args.seed , use_amp = args.use_amp , freeze_epochs = args.freeze_epochs )

if __name__ == "__main__":
    main()

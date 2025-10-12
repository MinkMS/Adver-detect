#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mục tiêu:
 - Huấn luyện mô hình ResNet18 từ đầu trên Food-101 (đã split).
 - Lưu lại mọi giá trị trong quá trình training: loss, acc, lr, thời gian, confusion matrix, classification report, v.v.
 - Lưu mô hình tốt nhất (val_acc cao nhất) và mô hình cuối cùng.
 - Log chi tiết từng epoch vào file CSV, confusion matrix và report vào npy/json.
 - Cấu trúc code rõ ràng, dễ bảo trì, dễ mở rộng.
"""

import os
import random
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets , transforms , models
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix , classification_report

# -------------------------
# CONFIG
# -------------------------
DATA_DIR = r"C:\Users\Mink\OneDrive\Documents\Coding\Datasets\Food-101\food-101-split"
BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_EPOCHS = 30
LR = 0.001
SEED = 4
DEVICE = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )

# Đường dẫn lưu model và logs
MODEL_DIR = r"C:\Users\Mink\OneDrive\Documents\Coding\GitHub\Adver-detect\resnet18"
LOG_DIR = os.path.join( MODEL_DIR , "logs" )
LOG_CSV = os.path.join( LOG_DIR , "resnet18_training_log.csv" )
BEST_MODEL_PATH = os.path.join( MODEL_DIR , "resnet18_food101_best.pth" )
FINAL_MODEL_PATH = os.path.join( MODEL_DIR , "resnet18_food101_final.pth" )
CONF_MATRIX_PREFIX = LOG_DIR
REPORT_JSON_PREFIX = LOG_DIR
REPORT_CSV_PREFIX = LOG_DIR
# -------------------------

# -------------------------
# Hàm tiện ích
# -------------------------
def set_seed( seed : int = 4 ):
    """
    Đặt seed cho random, numpy, torch để đảm bảo kết quả nhất quán.
    """
    torch.manual_seed( seed )
    np.random.seed( seed )
    random.seed( seed )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all( seed )
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_data_loaders( data_dir : str , batch_size : int , num_workers : int ):
    """
    Tạo các DataLoader cho train/val/test với các transform phù hợp.
    """
    data_transforms = {
        'train' : transforms.Compose( [
            transforms.RandomResizedCrop( 224 ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize( [ 0.485 , 0.456 , 0.406 ] , [ 0.229 , 0.224 , 0.225 ] )
        ] ),
        'val' : transforms.Compose( [
            transforms.Resize( 256 ),
            transforms.CenterCrop( 224 ),
            transforms.ToTensor(),
            transforms.Normalize( [ 0.485 , 0.456 , 0.406 ] , [ 0.229 , 0.224 , 0.225 ] )
        ] ),
        'test' : transforms.Compose( [
            transforms.Resize( 256 ),
            transforms.CenterCrop( 224 ),
            transforms.ToTensor(),
            transforms.Normalize( [ 0.485 , 0.456 , 0.406 ] , [ 0.229 , 0.224 , 0.225 ] )
        ] ),
    }
    image_datasets = { x : datasets.ImageFolder( os.path.join( data_dir , x ) , data_transforms[ x ] )
                      for x in [ 'train' , 'val' , 'test' ] }
    dataloaders = { x : torch.utils.data.DataLoader( image_datasets[ x ] , batch_size = batch_size ,
                                                  shuffle = ( x == 'train' ) , num_workers = num_workers )
                   for x in [ 'train' , 'val' , 'test' ] }
    dataset_sizes = { x : len( image_datasets[ x ] ) for x in [ 'train' , 'val' , 'test' ] }
    class_names = image_datasets[ 'train' ].classes
    return dataloaders , dataset_sizes , class_names

def build_resnet18( num_classes : int ):
    """
    Khởi tạo mô hình ResNet18 từ đầu, thay đổi lớp cuối cho đúng số lớp.
    """
    model = models.resnet18( weights = None )
    model.fc = nn.Linear( model.fc.in_features , num_classes )
    return model

def evaluate( model , dataloader , split_name , criterion , device , dataset_size ):
    """
    Đánh giá mô hình trên một tập dữ liệu, trả về loss, acc, confusion matrix, classification report.
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs , labels in dataloader:
            inputs = inputs.to( device )
            labels = labels.to( device )
            outputs = model( inputs )
            loss = criterion( outputs , labels )
            _ , preds = torch.max( outputs , 1 )
            running_loss += loss.item() * inputs.size( 0 )
            running_corrects += torch.sum( preds == labels.data ).item()
            all_labels.extend( labels.cpu().numpy() )
            all_preds.extend( preds.cpu().numpy() )
    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects / dataset_size
    cm = confusion_matrix( all_labels , all_preds )
    report = classification_report( all_labels , all_preds , output_dict = True , zero_division = 0 )
    return epoch_loss , epoch_acc , cm , report

# -------------------------
# Hàm huấn luyện chính
# -------------------------
def train_resnet18():
    """
    Huấn luyện ResNet18 trên Food-101, lưu lại mọi giá trị, mô hình tốt nhất và cuối cùng.
    """
    set_seed( SEED )
    # 1. Chuẩn bị dữ liệu
    dataloaders , dataset_sizes , class_names = get_data_loaders( DATA_DIR , BATCH_SIZE , NUM_WORKERS )
    num_classes = len( class_names )

    # 2. Khởi tạo mô hình, loss, optimizer, scheduler
    model = build_resnet18( num_classes ).to( DEVICE )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam( model.parameters() , lr = LR )
    scheduler = optim.lr_scheduler.StepLR( optimizer , step_size = 10 , gamma = 0.1 )

    # 3. Biến lưu log
    log_rows = []
    best_acc = 0.0
    best_model_wts = copy.deepcopy( model.state_dict() )

    # 4. Vòng lặp epoch
    for epoch in range( NUM_EPOCHS ):
        epoch_start = time.time()
        print( f"Epoch { epoch + 1 } / { NUM_EPOCHS }" )
        print( "-" * 10 )

        # --- Train ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        train_labels = []
        train_preds = []
        for inputs , labels in dataloaders[ 'train' ]:
            inputs = inputs.to( DEVICE )
            labels = labels.to( DEVICE )
            optimizer.zero_grad()
            outputs = model( inputs )
            loss = criterion( outputs , labels )
            _ , preds = torch.max( outputs , 1 )
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size( 0 )
            running_corrects += torch.sum( preds == labels.data ).item()
            train_labels.extend( labels.cpu().numpy() )
            train_preds.extend( preds.cpu().numpy() )
        train_loss = running_loss / dataset_sizes[ 'train' ]
        train_acc = running_corrects / dataset_sizes[ 'train' ]
        train_cm = confusion_matrix( train_labels , train_preds )
        train_report = classification_report( train_labels , train_preds , output_dict = True , zero_division = 0 )

        # --- Validation ---
        val_loss , val_acc , val_cm , val_report = evaluate( model , dataloaders[ 'val' ] , 'val' , criterion , DEVICE , dataset_sizes[ 'val' ] )

        # --- Test ---
        test_loss , test_acc , test_cm , test_report = evaluate( model , dataloaders[ 'test' ] , 'test' , criterion , DEVICE , dataset_sizes[ 'test' ] )

        # --- Log ---
        current_lr = optimizer.param_groups[ 0 ][ 'lr' ]
        log_rows.append( {
            'epoch' : epoch + 1 ,
            'train_loss' : train_loss ,
            'train_acc' : train_acc ,
            'val_loss' : val_loss ,
            'val_acc' : val_acc ,
            'test_loss' : test_loss ,
            'test_acc' : test_acc ,
            'lr' : current_lr ,
            'train_cm' : train_cm.tolist() ,
            'val_cm' : val_cm.tolist() ,
            'test_cm' : test_cm.tolist() ,
            'train_report' : train_report ,
            'val_report' : val_report ,
            'test_report' : test_report ,
            'time_sec' : time.time() - epoch_start
        } )

        print( f"Train Loss: { train_loss : .4f } Acc: { train_acc : .4f }" )
        print( f"Val   Loss: { val_loss : .4f } Acc: { val_acc : .4f }" )
        print( f"Test  Loss: { test_loss : .4f } Acc: { test_acc : .4f }" )
        print( f"LR: { current_lr : .6f } | Time: { log_rows[ -1 ][ 'time_sec' ] : .2f }s" )

        # --- Lưu mô hình tốt nhất ---
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy( model.state_dict() )
            torch.save( best_model_wts , BEST_MODEL_PATH )
            print( f"Best model saved at epoch { epoch + 1 } ( val_acc = { val_acc : .4f } )" )

        scheduler.step()

    # 5. Lưu log, mô hình, confusion matrix, report
    df = pd.DataFrame( log_rows )
    df.to_csv( LOG_CSV , index = False )
    print( f"Training log saved to { LOG_CSV }" )

    torch.save( model.state_dict() , FINAL_MODEL_PATH )
    print( f"Final model saved to { FINAL_MODEL_PATH }" )

    np.save( os.path.join( CONF_MATRIX_PREFIX , "train_confusion_matrices.npy" ) , np.array( [ row[ 'train_cm' ] for row in log_rows ] ) )
    np.save( os.path.join( CONF_MATRIX_PREFIX , "val_confusion_matrices.npy" ) , np.array( [ row[ 'val_cm' ] for row in log_rows ] ) )
    np.save( os.path.join( CONF_MATRIX_PREFIX , "test_confusion_matrices.npy" ) , np.array( [ row[ 'test_cm' ] for row in log_rows ] ) )
    import json
    with open( os.path.join( REPORT_JSON_PREFIX , "train_reports.json" ) , "w" ) as f:
        json.dump( [ row[ 'train_report' ] for row in log_rows ] , f )
    with open( os.path.join( REPORT_JSON_PREFIX , "val_reports.json" ) , "w" ) as f:
        json.dump( [ row[ 'val_report' ] for row in log_rows ] , f )
    with open( os.path.join( REPORT_JSON_PREFIX , "test_reports.json" ) , "w" ) as f:
        json.dump( [ row[ 'test_report' ] for row in log_rows ] , f )

    # Lưu các classification report ra file CSV
    def save_reports_csv( reports , filename_prefix ):
        all_rows = []
        for epoch_idx , report in enumerate( reports ):
            for label , metrics in report.items():
                if isinstance( metrics , dict ):
                    row = { 'epoch' : epoch_idx + 1 , 'label' : label }
                    row.update( metrics )
                    all_rows.append( row )
        df = pd.DataFrame( all_rows )
        df.to_csv( os.path.join( REPORT_CSV_PREFIX , f"{filename_prefix}_reports.csv" ) , index = False )

    save_reports_csv( [row['train_report'] for row in log_rows] , "train" )
    save_reports_csv( [row['val_report'] for row in log_rows] , "val" )
    save_reports_csv( [row['test_report'] for row in log_rows] , "test" )

    print( "All metrics saved." )

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    """
    Chạy script này để huấn luyện ResNet18 trên Food-101 đã split.
    Mọi giá trị sẽ được lưu lại để đánh giá sau.
    """
    train_resnet18()
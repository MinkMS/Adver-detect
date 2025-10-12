#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Mục tiêu:
 - Chia dataset dạng Food-101 (images / < class > / *.jpg) thành train / val / test
   với tỷ lệ chính xác (ví dụ 0.8 / 0.1 / 0.1).
 - Giữ stratified distribution nếu có scikit-learn; nếu không dùng thuật toán floor + distribute remainder.
 - Tạo manifest.csv ( filename , class , split , original_path ).
 - Hỗ trợ copy hoặc symlink, force, dry_run, và log summary.
"""

from __future__ import annotations
import argparse
import os
import csv
import tqdm
import math
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict , List , Tuple

# Optional progress bar
try:
    from tqdm import tqdm
    _HAVE_TQDM = True
except Exception:
    _HAVE_TQDM = False

# Try scikit-learn for stratified split
try:
    from sklearn.model_selection import StratifiedShuffleSplit
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False

# -------------------------
# CONFIG (edit here)
# -------------------------
# Default paths (edit these to match your environment)
INPUT_DIR = r"C:\Users\Mink\OneDrive\Documents\Coding\Datasets\Food-101\food-101"                # <-- change to your dataset root (contains 'images' folder)
OUTPUT_DIR = r"C:\Users\Mink\OneDrive\Documents\Coding\Datasets\Food-101\food-101-split"         # <-- change to desired output folder

# Default ratios (must sum to 1.0)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Other defaults
MIN_PER_SPLIT_PER_CLASS = 0    # best-effort minimum per class per split
USE_SYMLINK = False            # default copy behavior
SEED = 4
# -------------------------

def list_images_by_class( images_root : Path ) -> Dict[str , List[Path]]:
    """
    Scan images / < class > directories and return mapping class -> list of image Paths.
    """
    mapping : Dict[str , List[Path]] = {}
    for p in sorted( images_root.iterdir() ):
        if p.is_dir():
            imgs = [ f for f in sorted( p.iterdir() ) if f.is_file() ]
            if imgs:
                mapping[ p.name ] = imgs
    return mapping

def ensure_output_dirs( root : Path , splits : List[str] , classes : List[str] , force : bool = False ) -> None:
    """
    Create output folders for splits and classes. If force is True, remove existing output folder first.
    """
    if root.exists() and force:
        shutil.rmtree( root )
    root.mkdir( parents = True , exist_ok = True )
    for s in splits:
        for c in classes:
            ( root / s / c ).mkdir( parents = True , exist_ok = True )

def copy_or_link( src : Path , dst : Path , symlink : bool = False ) -> None:
    """
    Copy a file from src to dst or create a relative symlink (best-effort).
    """
    if dst.exists():
        return
    if symlink:
        try:
            rel = os.path.relpath( src , dst.parent )
            os.symlink( rel , dst )
        except Exception:
            shutil.copy2( src , dst )
    else:
        shutil.copy2( src , dst )

def split_with_sklearn( all_paths : List[Path] , all_labels : List[str] , train_count : int , val_count : int , test_count : int , seed : int ) -> Tuple[List[int] , List[int] , List[int]]:
    """
    Use StratifiedShuffleSplit in two steps to produce index lists for train / val / test.
    Returns ( train_idx , val_idx , test_idx ).
    """
    assert _HAVE_SKLEARN , "scikit-learn required for this function"
    # Step 1: train vs temp
    sss1 = StratifiedShuffleSplit( n_splits = 1 , train_size = train_count , random_state = seed )
    train_idx , temp_idx = next( sss1.split( all_paths , all_labels ) )
    # Step 2: split temp into val and test
    temp_labels = [ all_labels[i] for i in temp_idx ]
    temp_n = len( temp_idx )
    if temp_n == 0:
        val_idx = []
        test_idx = []
    else:
        val_fraction_of_temp = val_count / ( val_count + test_count ) if ( val_count + test_count ) > 0 else 0.0
        sss2 = StratifiedShuffleSplit( n_splits = 1 , train_size = val_fraction_of_temp , random_state = seed + 1 )
        val_rel_idx , test_rel_idx = next( sss2.split( temp_idx , temp_labels ) )
        val_idx = [ temp_idx[i] for i in val_rel_idx ]
        test_idx = [ temp_idx[i] for i in test_rel_idx ]
    return list( train_idx ) , list( val_idx ) , list( test_idx )

def compute_per_class_allocations( class_counts : Dict[str , int] , total_counts : Tuple[int , int , int] ) -> Dict[str , Tuple[int , int , int]]:
    """
    Compute per-class allocations ( train_i , val_i , test_i ) that sum to desired totals.
    Approach:
     - compute ideal fractional per class, take floors
     - distribute remaining slots by largest fractional remainders per split
     - ensure per-class sum <= class count
    """
    train_total , val_total , test_total = total_counts
    total_images = sum( class_counts.values() )
    if total_images == 0:
        return {}

    train_frac = train_total / total_images
    val_frac = val_total / total_images
    test_frac = test_total / total_images

    alloc = {}
    frac_list_train = []
    frac_list_val = []
    frac_list_test = []
    sum_train_floor = sum_val_floor = sum_test_floor = 0

    for cls , n in class_counts.items():
        ideal_t = n * train_frac
        ideal_v = n * val_frac
        ideal_te = n * test_frac
        floor_t = math.floor( ideal_t )
        floor_v = math.floor( ideal_v )
        floor_te = math.floor( ideal_te )
        sum_train_floor += floor_t
        sum_val_floor += floor_v
        sum_test_floor += floor_te
        frac_t = ideal_t - floor_t
        frac_v = ideal_v - floor_v
        frac_te = ideal_te - floor_te
        alloc[ cls ] = [ floor_t , floor_v , floor_te , frac_t , frac_v , frac_te ]
        frac_list_train.append( ( frac_t , cls ) )
        frac_list_val.append( ( frac_v , cls ) )
        frac_list_test.append( ( frac_te , cls ) )

    rem_train = train_total - sum_train_floor
    rem_val = val_total - sum_val_floor
    rem_test = test_total - sum_test_floor

    # Clip negatives (rare)
    if rem_train < 0:
        rem_train = 0
    if rem_val < 0:
        rem_val = 0
    if rem_test < 0:
        rem_test = 0

    # Distribute remaining slots by fractional remainder (largest first)
    frac_list_train.sort( reverse = True )
    frac_list_val.sort( reverse = True )
    frac_list_test.sort( reverse = True )

    i = 0
    while rem_train > 0:
        cls = frac_list_train[ i % len( frac_list_train ) ][ 1 ]
        alloc[ cls ][ 0 ] += 1
        rem_train -= 1
        i += 1

    i = 0
    while rem_val > 0:
        cls = frac_list_val[ i % len( frac_list_val ) ][ 1 ]
        alloc[ cls ][ 1 ] += 1
        rem_val -= 1
        i += 1

    i = 0
    while rem_test > 0:
        cls = frac_list_test[ i % len( frac_list_test ) ][ 1 ]
        alloc[ cls ][ 2 ] += 1
        rem_test -= 1
        i += 1

    # Ensure per-class sum <= n
    for cls , arr in alloc.items():
        total_assigned = arr[ 0 ] + arr[ 1 ] + arr[ 2 ]
        n = class_counts[ cls ]
        if total_assigned > n:
            excess = total_assigned - n
            # remove from splits with smallest fractional remainder
            fracs = [ ( arr[ 3 ] , 0 ) , ( arr[ 4 ] , 1 ) , ( arr[ 5 ] , 2 ) ]
            fracs_sorted = sorted( fracs , key = lambda x : x[ 0 ] )
            j = 0
            while excess > 0 and j < 3:
                split_idx = fracs_sorted[ j ][ 1 ]
                if alloc[ cls ][ split_idx ] > 0:
                    alloc[ cls ][ split_idx ] -= 1
                    excess -= 1
                else:
                    j += 1

    final_alloc : Dict[str , Tuple[int , int , int]] = { cls : ( arr[ 0 ] , arr[ 1 ] , arr[ 2 ] ) for cls , arr in alloc.items() }
    return final_alloc

def perform_manual_split( mapping : Dict[str , List[Path]] , train_count : int , val_count : int , test_count : int , seed : int = 4 ) -> Dict[str , str]:
    """
    mapping: class -> list of Paths
    returns dict path_str -> split ("train" / "val" / "test")
    """
    random.seed( seed )
    class_counts = { cls : len( lst ) for cls , lst in mapping.items() }
    total = sum( class_counts.values() )
    assert total == train_count + val_count + test_count , "Totals mismatch"

    per_class_alloc = compute_per_class_allocations( class_counts , ( train_count , val_count , test_count ) )
    assignment : Dict[str , str] = {}

    for cls , imgs in mapping.items():
        imgs_copy = imgs[:]
        random.shuffle( imgs_copy )
        alloc = per_class_alloc.get( cls , ( 0 , 0 , 0 ) )
        tcnt , vcnt , tscnt = alloc
        idx = 0
        for _ in range( tcnt ):
            if idx < len( imgs_copy ):
                assignment[ str( imgs_copy[ idx ] ) ] = "train"
                idx += 1
        for _ in range( vcnt ):
            if idx < len( imgs_copy ):
                assignment[ str( imgs_copy[ idx ] ) ] = "val"
                idx += 1
        for _ in range( tscnt ):
            if idx < len( imgs_copy ):
                assignment[ str( imgs_copy[ idx ] ) ] = "test"
                idx += 1
        # leftover -> train
        while idx < len( imgs_copy ):
            assignment[ str( imgs_copy[ idx ] ) ] = "train"
            idx += 1

    return assignment

# -------------------------
# Orchestrator
# -------------------------
def prepare_exact_splits( input_dir : Path , output_dir : Path , train_ratio : float , val_ratio : float , test_ratio : float , symlink : bool , min_per_split_per_class : int , seed : int , force : bool , dry_run : bool ) -> None:
    images_dir = input_dir / "images"
    if not images_dir.exists():
        raise FileNotFoundError( f"Images folder not found: { images_dir }" )

    mapping : Dict[str , List[Path]] = {}
    for p in sorted( images_dir.iterdir() ):
        if p.is_dir():
            imgs = [ f for f in sorted( p.iterdir() ) if f.is_file() ]
            if imgs:
                mapping[ p.name ] = imgs

    all_classes = sorted( list( mapping.keys() ) )
    if not all_classes:
        raise RuntimeError( "No classes found in images folder" )

    all_paths : List[Path] = []
    all_labels : List[str] = []
    for cls , imgs in mapping.items():
        for p in imgs:
            all_paths.append( p )
            all_labels.append( cls )

    total_images = len( all_paths )
    if total_images == 0:
        raise RuntimeError( "No images found" )

    train_count = int( round( total_images * train_ratio ) )
    val_count = int( round( total_images * val_ratio ) )
    test_count = total_images - train_count - val_count

    # Fix rounding drift
    if test_count < 0:
        diff = - test_count
        dec = min( diff , train_count )
        train_count -= dec
        diff -= dec
        if diff > 0:
            dec2 = min( diff , val_count )
            val_count -= dec2
            diff -= dec2
        test_count = total_images - train_count - val_count

    print( f"Using input: { input_dir }" )
    print( f"Using output: { output_dir }" )
    print( f"Total images: { total_images } ; desired -> train: { train_count } , val: { val_count } , test: { test_count }" )

    if min_per_split_per_class > 0:
        for cls , imgs in mapping.items():
            if len( imgs ) < min_per_split_per_class * 3:
                print( f"Warning: class '{ cls }' has only { len( imgs ) } images; MIN_PER_SPLIT_PER_CLASS may be impossible." , file = sys.stderr )

    assignment : Dict[str , str] = {}
    if _HAVE_SKLEARN:
        print( "scikit-learn detected: using StratifiedShuffleSplit." )
        train_idx , val_idx , test_idx = split_with_sklearn( all_paths , all_labels , train_count , val_count , test_count , seed )
        for i in train_idx:
            assignment[ str( all_paths[ i ] ) ] = "train"
        for i in val_idx:
            assignment[ str( all_paths[ i ] ) ] = "val"
        for i in test_idx:
            assignment[ str( all_paths[ i ] ) ] = "test"
        # any unassigned -> train
        for i in range( len( all_paths ) ):
            key = str( all_paths[ i ] )
            if key not in assignment:
                assignment[ key ] = "train"
    else:
        print( "scikit-learn not found: using manual allocation." )
        assignment = perform_manual_split( mapping , train_count , val_count , test_count , seed )

    counts = Counter( assignment.values() )
    print( "Actual split counts:" , counts )

    splits = [ "train" , "val" , "test" ]
    ensure_output_dirs( output_dir , splits , all_classes , force = force )

    ops : List[Tuple[Path , Path]] = []
    it = assignment.items()
    if _HAVE_TQDM:
        it = tqdm( assignment.items() , desc = "Copying / linking" )
    for path_str , split in it:
        src = Path( path_str )
        if not src.exists():
            # try resolve under images folder by filename
            fname = src.name
            found = None
            for cls in all_classes:
                cand = images_dir / cls / fname
                if cand.exists():
                    found = cand
                    break
            if found:
                src = found
            else:
                print( f"Warning: source not found: { path_str }" )
                continue
        cls = src.parent.name
        dst = output_dir / split / cls / src.name
        if dry_run:
            ops.append( ( src , dst ) )
        else:
            copy_or_link( src , dst , symlink = symlink )

    manifest_path = output_dir / "manifest.csv"
    if not dry_run:
        with manifest_path.open( 'w' , newline = '' , encoding = 'utf-8' ) as mf:
            writer = csv.writer( mf )
            writer.writerow( [ "filename" , "class" , "split" , "original_path" ] )
            for path_str , split in assignment.items():
                p = Path( path_str )
                fname = p.name
                cls = p.parent.name
                writer.writerow( [ fname , cls , split , str( p ) ] )
        print( f"Manifest saved to { manifest_path }" )
    else:
        print( "Dry-run: no manifest written." )

    print( "Done." )

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser( description = "Split Food-101 dataset into exact train / val / test proportions." )
    p.add_argument( "--input" , "-i" , default = INPUT_DIR , help = "Path to dataset root (contains 'images' folder)." )
    p.add_argument( "--output" , "-o" , default = OUTPUT_DIR , help = "Output folder for splits." )
    p.add_argument( "--train" , type = float , default = TRAIN_RATIO , help = "Train ratio (sum must be 1.0)." )
    p.add_argument( "--val" , type = float , default = VAL_RATIO , help = "Val ratio." )
    p.add_argument( "--test" , type = float , default = TEST_RATIO , help = "Test ratio." )
    p.add_argument( "--symlink" , action = "store_true" , default = USE_SYMLINK , help = "Create symlinks instead of copying." )
    p.add_argument( "--min_per_split" , type = int , default = MIN_PER_SPLIT_PER_CLASS , help = "Best-effort min items per class per split." )
    p.add_argument( "--seed" , type = int , default = SEED , help = "Random seed." )
    p.add_argument( "--force" , action = "store_true" , help = "Remove output folder if exists." )
    p.add_argument( "--dry_run" , action = "store_true" , help = "Simulate only; do not copy / link or write manifest." )
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    s = args.train + args.val + args.test
    if abs( s - 1.0 ) > 1e-6:
        print( "Error: train + val + test ratios must sum to 1.0" , file = sys.stderr )
        sys.exit( 1 )
    inp = Path( args.input ).expanduser().resolve()
    outp = Path( args.output ).expanduser().resolve()
    prepare_exact_splits( inp , outp , args.train , args.val , args.test , args.symlink , args.min_per_split , args.seed , args.force , args.dry_run )

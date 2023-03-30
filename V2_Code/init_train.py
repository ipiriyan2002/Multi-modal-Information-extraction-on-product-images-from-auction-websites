from utils import load_voc_2007
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Download Pascal 2007 Dataset')
    parser.add_argument("--target-dir", help="target directory for dataset")
    parser.add_argument("--split", help="the split to download")
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parse_args()
    tdir = "./Data/"
    split = "train"
    
    if args.target_dir:
        tdir = args.target_dir
    if args.split:
        split = args.split
    
    load_voc_2007(target_dir=tdir, split=split)
    
    

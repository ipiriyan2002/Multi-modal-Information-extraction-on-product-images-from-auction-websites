#Import custom packages
from Utils.utils import load_voc_2007
#Import other packages
import argparse
import os

def parse_args():
    """
    Argument parser
    Return:
        args : parsed arguments
    """
    parser = argparse.ArgumentParser(description='Download Pascal 2007 Dataset')
    parser.add_argument("--target-dir", default="./data/",help="target directory for dataset")
    parser.add_argument("--split", default="train",help="the split to download")
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parse_args()
    tdir = args.target_dir
    split = args.split
    
    #Download the voc 2007 dataset
    load_voc_2007(target_dir=tdir, split=split)
    
    

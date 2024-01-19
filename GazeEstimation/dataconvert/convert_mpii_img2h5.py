"""
Convert eye jpg images from the src_dir to h5 (dst_file)
Default output image size: 64x64
Output image channel: grayscale (fixed)
"""

import argparse
from glob import glob
import h5py
import os

from data_converter import DataConverter


parser = argparse.ArgumentParser()

# source data directory
parser.add_argument(
    "--src_dir",
    type=str,
    default="data_dir",
    help="source directory (assuming the directory is in GazeML/datasets",
)

# destination h5 file
parser.add_argument(
    "--dst_file",
    type=str,
    default="testh5.h5",
    help="destination h5 file (assuming the file will be in GazeML/datasets",
)

# type of the dataset
parser.add_argument(
    "--type", type=str, default='train', help="type of the dataset - train or test"
)

# image size
parser.add_argument("--width", type=int, default=64, help="image width")

parser.add_argument("--height", type=int, default=64, help="image height")

# metadata
parser.add_argument("--n_people", type=int, default=15, help="number of participants")

# write mode
# w - create file, truncate if exists
# w- - create file, fail if exists
parser.add_argument("--mode", type=str, default="w", help="h5 file write mode")


if __name__ == "__main__":
    # Parameters
    params = parser.parse_args()

    img_paths = sorted(glob(os.path.join(params.src_dir, "*.jpg")))
    (ow, oh) = (params.width, params.height)

    converter = DataConverter(
        path=params.dst_file,
        mode=params.mode,
        imsize=(ow, oh),
        n_people=params.n_people
    )
    if params.type == 'test':
        converter.write_data(subset_name='test', img_paths=img_paths)
    else:
        converter.write_data(subset_name='train', img_paths=img_paths)

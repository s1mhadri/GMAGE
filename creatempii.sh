#!/bin/bash

cp -r /projects/holagundhi/GISED/GazeRedirection/eval/mpiigenesT /projects/holagundhi/Datasets/gised_datasets/mpiigenall

cp -r /projects/holagundhi/GISED/GazeRedirection/eval/mpiigenes /projects/holagundhi/Datasets/gised_datasets/mpiiall

cp -r /projects/holagundhi/GISED/GazeRedirection/eval/mpiigenesT /projects/holagundhi/Datasets/gised_datasets/mpiiall

cp -r /projects/holagundhi/Datasets/dataset_gr/mpiigaze /projects/holagundhi/Datasets/gised_datasets/mpiiall

cd /projects/holagundhi/GISED/GazeEstimation/dataconvert

python convert_mpii_img2h5.py --src_dir /projects/holagundhi/Datasets/gised_datasets/mpiigenall --dst_file /projects/holagundhi/Datasets/gised_datasets/h5files/mpiigenall.h5

python convert_mpii_img2h5.py --src_dir /projects/holagundhi/Datasets/gised_datasets/mpiiall --dst_file /projects/holagundhi/Datasets/gised_datasets/h5files/mpiiall.h5
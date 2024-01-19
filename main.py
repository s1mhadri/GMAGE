import os
import glob
import shutil
import subprocess
import sys
from math import floor
from distutils.dir_util import copy_tree

### Uncomment the part you want to run ###
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 4, 5"
# -------------------------------------------------------------------------------------
# # Root Directories
# -------------------------------------------------------------------------------------

# root directory for gaze redirection code
root_gr_dir = "/projects/holagundhi/GISED/GazeRedirection"
# root directory for gaze estimation code
root_ge_dir = "/projects/holagundhi/GISED/GazeEstimation"
# root directory for all dataset used for training
root_dataset_path = "/projects/holagundhi/Datasets"
# h5 files directory
h5files_dir = os.path.join(root_dataset_path, "gised_datasets/h5files")

# -------------------------------------------------------------------------------------
# # k-fold cross-validation params
# -------------------------------------------------------------------------------------

# n_folds = 5
# fold_size = floor(56 / n_folds)

# #####################################################################################
# # Train and Generate Redirected Images
# #####################################################################################

# log directory for gaze redirection
# log_gr_dir = os.path.join(root_gr_dir, 'log')

# -------------------------------------------------------------------------------------
# # Train gaze redirection model
# -------------------------------------------------------------------------------------

# dataset path for training gaze redirection model
all_data_path = os.path.join(root_dataset_path, 'dataset_gr/all')
# vgg model path for training gaze redirection model
vgg_path = os.path.join(root_gr_dir, 'vgg_16.ckpt')

# os.chdir(root_gr_dir)
# for fold_id in range(5):
# log_gr_dir = os.path.join(root_gr_dir, 'log_col')
# subprocess.run(
#     [
#         "python",
#         "main_gr.py",
#         "--mode",
#         "train",
#         "--epochs",
#         "20",
#         "--data_path",
#         all_data_path,
#         "--log_dir",
#         log_gr_dir,
#         "--batch_size",
#         "32",
#         "--vgg_path",
#         vgg_path,
#     ]
# )

# -------------------------------------------------------------------------------------
# # Generate redirected images
# -------------------------------------------------------------------------------------

# directory containing dataset for gaze redirection
dataset_gr_dir = os.path.join(root_dataset_path, 'dataset_gr')
eval_data_path = os.path.join(dataset_gr_dir, 'all')
# for fold_id in range(5):
log_gr_dir = os.path.join(root_gr_dir, 'log_mpii')
# os.chdir(root_gr_dir)
# subprocess.run(
#     [
#         "python",
#         "main_gr.py",
#         "--mode",
#         "eval",
#         "--data_path",
#         eval_data_path,
#         "--log_dir",
#         log_gr_dir,
#         "--batch_size",
#         "32",
#     ]
# )

# #####################################################################################
# # Dataset Preparation
# #####################################################################################

# root directory for all dataset used for training
dataset_dir = os.path.join(root_dataset_path, 'gised_datasets')

# -------------------------------------------------------------------------------------
# # Flip horizontal angles of right eye images of columbia original dataset
# -------------------------------------------------------------------------------------

# os.chdir(os.path.join(dataset_dir, 'columbia_orig'))
# n = 0 # sequence number to prevent deleting duplicates

# for img_name in img_names:
#     if 'R' in img_name:
#         img_label = img_name.split('_')
#         for i, label in enumerate(img_label):
#             if 'H' in label:
#                 yaw = int(label.strip('H'))
#                 yaw *= -1
#                 img_label[i] = '%dH' %(yaw)
#             if '.jpg' in label:
#                 new_label = label.strip('.jpg') + '_%d.jpg' %(n)
#                 img_label[i] = new_label
#                 n += 1
#         os.rename(img_name, '_'.join(img_label))

# -------------------------------------------------------------------------------------
# # Combine columbia original images with generated images
# -------------------------------------------------------------------------------------

# generated images directory
eval_dir = os.path.join(root_gr_dir, 'log/eval')

# def copy_files(src='', dst=''):
#     files = glob.glob(src+'/*.jpg')
#     for f in files:
#         shutil.copy(f, dst)

# source = os.path.join(dataset_dir, 'columbia_orig')
# destination = os.path.join(dataset_dir, 'gen_all')
# if not os.path.exists(destination):
#     os.makedirs(destination)
# # copy_files(source, destination)

# for idir in sorted(os.listdir(eval_dir)):
#     source2 = os.path.join(eval_dir, idir)
#     copy_files(source2, destination)

# -------------------------------------------------------------------------------------
# # Rename index of generated images
# -------------------------------------------------------------------------------------
# n = 0
# os.chdir(os.path.join(dataset_dir, 'gen_all'))
# img_names = sorted(glob.glob('*.jpg'))
# for image in img_names:
#     img_label = image.split('_')
#     img_label[1] = '%d' %n
#     n += 1
#     os.rename(image, '_'.join(img_label))

# print(n)

# -------------------------------------------------------------------------------------
# # calculate number of images in each subdirectory of a given directory
# -------------------------------------------------------------------------------------

# imgdir = dataset_gr_dir
# for idir in sorted(os.listdir(imgdir)):
#     files = glob.glob(os.path.join(imgdir, idir)+'/*.jpg')
#     print(idir, '--', len(files))

# -------------------------------------------------------------------------------------
# # calculate number of images of given type and copy them to given folder
# -------------------------------------------------------------------------------------

# i = 0
# # imgdir = os.path.join(dataset_gr_dir, '0P')
# imgdir = os.path.join(dataset_dir, 'all')
# HV5dir = os.path.join(dataset_dir, 'colmatrix')
# # if not os.path.exists(HV5dir):
# #     os.makedirs(HV5dir)
# os.chdir(imgdir)
# # files = sorted(glob.glob(imgdir+'/*.jpg'))
# files = sorted(glob.glob('*.jpg'))
# for file in files:
#     lab = file.split('_')
#     # if 'H' in lab[-2]:
#     #     lab[-2], lab[-3] = lab[-3], lab[-2]
    
#     # valH = abs(float(file.split('/')[-1].split('_')[-3].strip('H')))
#     # valV = abs(float(file.split('/')[-1].split('_')[-2].strip('V')))
#     valH = abs(float(lab[-2].strip('H')))
#     valV = abs(float(lab[-3].strip('V')))
#     print(valH, valV)

# print(i)

# -------------------------------------------------------------------------------------
# # Convert dataset directory to h5 file
# -------------------------------------------------------------------------------------

# directory for data converter code
dataconvert_dir = os.path.join(root_ge_dir, 'dataconvert')
src_dir = os.path.join(root_gr_dir, "eval", "col2mpii")
dst_file =  os.path.join(h5files_dir, 'col2mpii.h5')

os.chdir(dataconvert_dir)
# for i in range(2):
#     for j in range(2):
#         imgdir1 = os.path.join(dataset_gr_dir, 'colmatrix'+str(i)+str(j))
#         imgdir2 = os.path.join(eval_dir, 'matrix'+str(i)+str(j))
#         dstdir = os.path.join(dataset_dir, 'allmatrix'+str(i)+str(j))
#         copy_tree(imgdir1, dstdir)
#         copy_tree(imgdir2, dstdir)
#         src_dir = dstdir
#         dst_file =  os.path.join(h5files_dir, 'allmatrix'+str(i)+str(j)+'.h5')

# subprocess.run(
#     [
#         'python',
#         'convert_mpii_img2h5.py',
#         '--src_dir',
#         src_dir,
#         '--dst_file',
#         dst_file,
#         '--type',
#         'train',
#     ]
# )

# #####################################################################################
# # Train and evaluate DPG Gaze estimation
# #####################################################################################

# train and test dataset paths
train_h5file = 'col_orig.h5'
train_path = os.path.join(h5files_dir, train_h5file)
test_h5file = 'mpii_orig.h5'
test_path = os.path.join(h5files_dir, test_h5file)

# log directory for gaze estimation - '/projects/holagundhi/GISED/outputs'
log_ge_dir = 'outputs/col2mpii_orig'

os.chdir(root_ge_dir)
subprocess.run(
    [
        "python",
        "main_mpii_ge.py",
        "--train_path",
        train_path,
        "--test_path",
        test_path,
        "--epoch",
        "20",
        "--n_train",
        "56",
        "--n_test",
        "15",
        "--log_dir",
        log_ge_dir,
    ]
)


# for i in range(2):
#     for j in range(2):
#         train_h5file = 'colmatrix'+str(i)+str(j)+'.h5'
#         log_ge_dir = 'outputs/out_col2matrix'+str(i)+str(j)
#         train_path = os.path.join(h5files_dir, train_h5file)
#         subprocess.run(
#             [
#                 "python",
#                 "main_ge.py",
#                 "--train_path",
#                 train_path,
#                 "--test_path",
#                 test_path,
#                 "--epoch",
#                 "90",
#                 "--log_dir",
#                 log_ge_dir,
#             ]
#         )
# -------------------------------------------------------------------------------------

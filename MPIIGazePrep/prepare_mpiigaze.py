import os
from glob import glob
import shutil
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


label_files = glob('./Label/p??.label')
img_dir = './Image'
heads = []
root_dataset_path = "/projects/holagundhi/Datasets"
output_dir = os.path.join(root_dataset_path, 'dataset_gr', 'mpiigaze')

if os.path.isdir(output_dir):
    shutil.rmtree(output_dir)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)


for label_path in sorted(label_files):
    with open(label_path) as f:
        labels = f.read().splitlines()[1:]
        labels = [i.strip().split(' ') for i in labels]
    
    img_paths = [i[0] for i in labels]
    whichEyes = [i[2] for i in labels]
    gaze_angles = np.array([list(map(float, i[5].split(','))) for i in labels])
    head_angles = np.array([list(map(float, i[6].split(','))) for i in labels])
    gaze_angles = (gaze_angles / np.pi * 180)
    head_angles = (head_angles / np.pi * 180)
    
    for i, img_path in enumerate(tqdm(img_paths)):
        ids = int(label_path.split('/')[-1].strip('p.label'))
        dist = 'NaNm'  # distance to screen in the Columbia dataset. Unknown for MPIIGaze
        head = round(head_angles[i,0], 1)  # head yaw angle
        yaw = round(gaze_angles[i, 0], 2)  # yaw gaze angle
        pitch = round(gaze_angles[i, 1], 2)  # pitch gaze angle
        whichEye = 'L' if whichEyes[i] == 'left' else 'R'
        if head == 0:
            head = 0
        # if 0.001 > head > -0.001:
        #     head = 0
        # heads.append(head)
        old_img_name = img_path.split('/')[-1]
        old_img_path = os.path.join(img_dir, img_path)
        new_img_path = os.path.join(output_dir, old_img_name)
        new_img_name = os.path.join(
            output_dir,
            '%04d_%04d%s_%.2fP_%.2fV_%.2fH_%s.jpg' % (
            (ids+1), i, 'm', head, yaw, pitch, whichEye)
        )
        # if head == 0:
        #     print(new_img_name)
        #     x += 1
    # print(x)
    # break
        shutil.copyfile(old_img_path, new_img_path)
        os.rename(new_img_path, new_img_name)

# heads = np.array(heads)
# q25, q75 = np.percentile(heads, [25, 75])
# bin_width = 2 * (q75 - q25) * len(heads) ** (-1/3)
# bins = round((heads.max() - heads.min()) / bin_width)

# sns_plot = sns.displot(heads, bins=bins, kde=True)
# sns_plot.set(
#     title='Histogram of Head angles',
#     xlabel='Angles in degrees',
#     ylabel='Number of images'
# )
# sns_plot.savefig("./head_ints.png") 

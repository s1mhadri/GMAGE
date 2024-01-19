import numpy as np
import scipy.io as sio
import cv2
import os
from glob import glob
import shutil
from tqdm import tqdm

import data_processing_core as dpc


dataset_path = '/projects/holagundhi/Datasets/MPIIGaze'
sample_root = os.path.join(dataset_path, 'Evaluation Subset/sample list for eye image')
out_root = "."
scale = True
img_scale = 1
imsize = (int(64*img_scale), int(64*img_scale)) # MODIFICATION

def AnnoDecode(anno_info):
	annotation = np.array(anno_info.strip().split(" ")).astype("float32")
	out = {}
	out["left_left_corner"] = annotation[0:2]
	out["left_right_corner"] = annotation[6:8]
	out["right_left_corner"] = annotation[12:14]
	out["right_right_corner"] = annotation[18:20]
	out["headrotvectors"] = annotation[29:32]
	out["headtransvectors"] = annotation[32:35]
	out["rightcenter"] = annotation[35:38]
	out["leftcenter"] = annotation[38:41]
	out["target"] = annotation[26:29]
	return out

def ImageProcessing_Person(im_root, sample_list, im_outpath, label_outpath, person):
    days = sorted(os.listdir(im_root))
    days.remove("Calibration")

    # Read camera matrix
    camera = sio.loadmat(os.path.join(f"{im_root}", "Calibration", "Camera.mat"))
    camera = camera["cameraMatrix"]

    # Read gaze annotation
    anno_dict = {}
    for day in days:
        day_path = os.path.join(im_root, day)
        annotation = os.path.join(day_path, "annotation.txt")

        with open(annotation) as annfile:
            anno_mes = annfile.readlines()
        
        anno_dict[day] = anno_mes
    
    # Create the handle of label 
    outfile = open(label_outpath, 'w')
    outfile.write("Image Original WhichEye 3DGaze 3DHead 2DGaze 2DHead Rmat Smat GazeOrigin\n")

    # Image Processing 
    with open(sample_list) as infile:
        im_list = infile.readlines()
        total = len(im_list)
    
    for count, info in enumerate(tqdm(im_list)):
        # Read image info
        im_info, which_eye = info.strip().split(" ")
        day, im_name = im_info.split("/")
        im_number = int(im_name.split(".")[0])

        # Read image annotation and image
        im_path = os.path.join(im_root, day, im_name)
        im = cv2.imread(im_path, cv2.IMREAD_COLOR) # MODIFICATION
        annotation = anno_dict[day][im_number-1]
        annotation = AnnoDecode(annotation)

        # Normalize the image
        if which_eye == "left":
            norm = dpc.norm(center = annotation["leftcenter"],
                            gazetarget = annotation["target"],
                            headrotvec = annotation["headrotvectors"],
                            imsize = imsize,
                            camparams = camera)
            origin = norm.GetCoordinate(annotation["leftcenter"])
     
        else:
            norm = dpc.norm(center = annotation["rightcenter"],
                            gazetarget = annotation["target"],
                            headrotvec = annotation["headrotvectors"],
                            imsize = imsize,
                            camparams = camera)
            origin = norm.GetCoordinate(annotation["rightcenter"])
        
        # Acquire essential info
        im_eye = norm.GetImage(im)
        im_eye = cv2.cvtColor(im_eye, cv2.COLOR_BGR2YUV)
        im_eye[:,:,0] = cv2.equalizeHist(im_eye[:,:,0]) # equalize the y channel
        im_eye = cv2.cvtColor(im_eye, cv2.COLOR_YUV2BGR)

        gaze = norm.GetGaze(scale=scale)
        head = norm.GetHeadRot(vector=True)

        if which_eye == "right":
            im_eye = cv2.flip(im_eye, 1)
            gaze = dpc.GazeFlip(gaze)
            head = dpc.HeadFlip(head) 
            origin[0] = -origin[0]
        
        gaze_2d = dpc.GazeTo2d(gaze)
        head_2d = dpc.HeadTo2d(head)

        rvec, svec = norm.GetParams()

        # Save the acquired info
        cv2.imwrite(os.path.join(im_outpath, str(count+1)+".jpg"), im_eye)
        
        save_name = os.path.join(person, str(count+1) + ".jpg")
        save_origin = im_info
        save_flag = which_eye
        save_gaze = ",".join(gaze.astype("str"))
        save_head = ",".join(head.astype("str"))
        save_gaze2d = ",".join(gaze_2d.astype("str"))
        save_head2d = ",".join(head_2d.astype("str"))
        save_rvec = ",".join(rvec.astype('str')) 
        save_svec = ",".join(svec.astype('str')) 
        origin = ",".join(origin.astype('str')) 
        
        save_str = " ".join([save_name, save_origin, save_flag, save_gaze, save_head, save_gaze2d, save_head2d, save_rvec, save_svec, origin])
        
        outfile.write(save_str + "\n")

    outfile.close()

def ImageProcessing_MPII():
    persons = sorted(os.listdir(sample_root))
    for person in persons:
        sample_list = os.path.join(sample_root, person)
        person = person.split(".")[0]  # p00, p01, ...
        im_root = os.path.join(dataset_path, "Data", "Original", person)
        im_outpath = os.path.join(out_root, "Image", person)
        label_outpath = os.path.join(out_root, "Label", f"{person}.label")

        if not os.path.exists(im_outpath):
            os.makedirs(im_outpath)
        if not os.path.exists(os.path.join(out_root, "Label")):
            os.makedirs(os.path.join(out_root, "Label"))
        
        print(f"Start Processing {person}")
        ImageProcessing_Person(im_root, sample_list, im_outpath, label_outpath, person)
        

if __name__ == "__main__":
    ImageProcessing_MPII()
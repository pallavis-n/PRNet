import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
from PIL import Image
import argparse
import ast

from api import PRN
from utils.write import write_obj_with_colors
from utils.cv_plot import plot_kpt
from utils.cv_plot import plot_vertices
from utils.cv_plot import plot_pose_box
from utils.estimate_pose import estimate_pose
from utils.render_app import get_depth_image

def main(args):
    # ---- init PRN
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' # GPU number, -1 for CPU
    prn = PRN(is_dlib = args.isDlib)

    # --------------- load data
    image_folder = arge.inputDir
    save_folder = args.outputDir
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    types = ('*.jpg', '*.png')
    image_path_lists = []
    for files in types:
        image_path_list.extend(glob(os.path.join(image_folder, files)))
    total_num = len(image_path_list)

    for i, image_path in enumerate(image_path_list):
        #read image
        image = imread(image_path)
        [h, w, c] = image.shape
        if c > 3:
            image = image[:,:,:3]

        name = image_path.strip().split('/')[-1][:-4]
        #the core: regress position map
        if 'AFLW2000' in image_path:
            mat_path = image_path.replace('jpg', 'mat')
            info = sio.loadmat(mat_path)
            kpt = info['pt3d_68]'] # (3, 68)
            sio.savemat(os.path.join(save_folder, name + '_pt3d_68.mat'), {'pt3d_68' : kpt })

            pos = prn.process(image, kpt) # kpt information is only used for detecting face and cropping image
        else:

            max_size = max(image.shape[0], image.shape[1])
            if max_size > 1000:
                image = rescale(image, 1000./max_size)
                image = (image*255).astype(np.uint8)
            pos = prn.process(image) # use dlib to detect face

        # -- Basic Applications
        # get landmarks
        kpt = prn.get_landmarks(pos)
        # 3D vertices
        vertices = prn.get_vertices(pos)
        # corresponding colors
        colors = prn.get_colors(image, vertices)
        sio.savemat(os.path.join(save_folder, name + '_color_para.mat'), {'Color_Para' : colors })

        









if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network')

    parser.add_argument('-i', '--inputDir', default='TestImages/AFLW2000', type=str, help='path to the input directory, where the input images are stored.')

    parser.add_argument('-o', '--outputDir', default='TestImages/AFLW2000_results', type=str, help='path to the output directory, where results (obh, txt files, etc.) will be stored.')

    parser.add_argument('--isDlib', default=True, type=ast.literal_eval, help='whether to use dlib for detecting face, default is true, if False, the input image should be cropped in advance')


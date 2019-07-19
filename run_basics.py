import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from time import time
from PIL import Image

from api import PRN
from utils.write import write_obj_with_colors
from utils.cv_plot import plot_kpt
from utils.cv_plot import plot_vertices
from utils.cv_plot import plot_pose_box
from utils.estimate_pose import estimate_pose
from utils.render_app import get_depth_image

# ---- init PRN
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # GPU number, -1 for CPU
prn = PRN(is_dlib = False) 


# ------------- load data
image_folder = 'TestImages/AFLW2000/'
save_folder = 'TestImages/AFLW2000_results'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

types = ('*.jpg', '*.png')
image_path_list= []
for files in types:
    image_path_list.extend(glob(os.path.join(image_folder, files)))
total_num = len(image_path_list)

for i, image_path in enumerate(image_path_list):
    # read image
    image = imread(image_path)
    [h, w, c] = image.shape

    # the core: regress position map    
    if 'AFLW2000' in image_path:
        mat_path = image_path.replace('jpg', 'mat')
        info = sio.loadmat(mat_path)
        kpt = info['pt3d_68'] # (3, 68))
        
        # get the keypoints and place them on the original 2D picture
        kpt_T = (np.transpose(kpt)) # transpose matrix to get points in (68, 3)
        kpt_pts = plot_kpt(image, kpt_T)
        kpt_img = Image.fromarray(kpt_pts)
        
        pos = prn.process(image, kpt) # kpt information is only used for detecting face and cropping image
    else:
        pos = prn.process(image) # use dlib to detect face

    # -- Basic Applications
    # get landmarks
    kpt = prn.get_landmarks(pos)
    # 3D vertices
    vertices = prn.get_vertices(pos)
    # corresponding colors
    colors = prn.get_colors(image, vertices)

    # find the mesh of 3D vertices and put it on the original 2D picture
    mesh = plot_vertices(image, vertices)
    mesh_plot = Image.fromarray(mesh)

    # find the 3D plot box and put it on the original 2D picture
    P, pose = estimate_pose(vertices) # use the function to get P (affine camera matrix) and pose direction
    box = plot_pose_box(image, P, kpt)
    box_plot = Image.fromarray(box)

    # get the depth image information and save to an image file to output later
    depth = get_depth_image(vertices, prn.triangles, h, w)
    depth_plot = Image.fromarray(depth)

    # -- save
    name = image_path.strip().split('/')[-1][:-4]
    np.savetxt(os.path.join(save_folder, name + '.txt'), kpt)
    kpt_img.save(os.path.join(save_folder, name + '_kpt.png')) # save the image in the same location as the others
    mesh_plot.save(os.path.join(save_folder, name + '_vertices_mesh.png')) # save the mesh image in same location
    box_plot.save(os.path.join(save_folder, name + '_pose_box.png')) # save the  box plot in same location
    depth_plot.save(os.path.join(save_folder, name + '_depth_image.gif')) # save the depth image in same location
    write_obj_with_colors(os.path.join(save_folder, name + '.obj'), vertices, prn.triangles, colors) #save 3d face(can open with meshlab)

    sio.savemat(os.path.join(save_folder, name + '_mesh.mat'), {'vertices': vertices, 'colors': colors, 'triangles': prn.triangles})

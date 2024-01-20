import torch
import numpy as np

from os import path as osp

support_dir = '../support_data/'
# Choose the device to run the body model on.
comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

amass_npz_fname = osp.join(support_dir, 'github_data/dmpl_sample.npz') # the path to body data
bdata = np.load(amass_npz_fname)

# you can set the gender manually and if it differs from data's then contact or interpenetration issues might happen
subject_gender = bdata['gender']

print('Data keys available:%s'%list(bdata.keys()))

print('The subject of the mocap sequence is  {}.'.format(subject_gender))
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

#from human_body_prior.body_model.body_model import BodyModel
#from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.src.human_body_prior.body_model.body_model import BodyModel
from human_body_prior.src.human_body_prior.tools.omni_tools import copy2cpu as c2c

if str(subject_gender)[:2]=="b'":
    subject_gender_final = str(subject_gender)[2:]
    subject_gender_final = subject_gender_final[:-1]
    subject_gender_final = np.array(subject_gender_final)
else:
    subject_gender_final = subject_gender

bm_fname = osp.join(support_dir, 'body_models/smplh/{}/model.npz'.format(subject_gender_final))
dmpl_fname = osp.join(support_dir, 'body_models/dmpls/{}/model.npz'.format(subject_gender_final))

num_betas = 16 # number of body parameters
num_dmpls = 8 # number of DMPL parameters

bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(comp_device)
faces = c2c(bm.f)

time_length = len(bdata['trans'])

body_parms = {
    'root_orient': torch.Tensor(bdata['poses'][:, :3]).to(comp_device), # controls the global root orientation
    'pose_body': torch.Tensor(bdata['poses'][:, 3:66]).to(comp_device), # controls the body
    'pose_hand': torch.Tensor(bdata['poses'][:, 66:]).to(comp_device), # controls the finger articulation
    'trans': torch.Tensor(bdata['trans']).to(comp_device), # controls the global body position
    'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=time_length, axis=0)).to(comp_device), # controls the body shape. Body shape is static
    'dmpls': torch.Tensor(bdata['dmpls'][:, :num_dmpls]).to(comp_device) # controls soft tissue dynamics
}

print('Body parameter vector shapes: \n{}'.format(' \n'.join(['{}: {}'.format(k,v.shape) for k,v in body_parms.items()])))
print('time_length = {}'.format(time_length))

import trimesh
from body_visualizer.tools.vis_tools import colors
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.mesh.sphere import points_to_spheres
from body_visualizer.tools.vis_tools import show_image
import cv2
from PIL import Image as im 

imw, imh=1600, 1600
mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

body_pose_beta = bm(**{k:v for k,v in body_parms.items() if k in ['pose_body', 'betas']})

def vis_body_pose_beta(fId = 0):
    body_mesh = trimesh.Trimesh(vertices=c2c(body_pose_beta.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    save_img_hope = im.fromarray(body_image) 
    save_img_hope.save('img_hope.jpg')
    #cv2.imwrite("/home/up201705948/amass-master", body_image)
    print(body_image)
    #show_image(body_image)

vis_body_pose_beta(fId=0)

body_trans_root = bm(**{k:v for k,v in body_parms.items() if k in ['pose_body', 'betas', 'pose_hand', 'dmpls',
                                                                   'trans', 'root_orient']})

def vis_body_trans_root(fId = 0):
    body_mesh = trimesh.Trimesh(vertices=c2c(body_trans_root.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    #show_image(body_image)
    save_img_hope = im.fromarray(body_image) 
    save_img_hope.save('img_hope1.jpg')

vis_body_trans_root(fId=0)

def vis_body_transformed(fId = 0):
    body_mesh = trimesh.Trimesh(vertices=c2c(body_trans_root.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
    body_mesh.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 1)))
    body_mesh.apply_transform(trimesh.transformations.rotation_matrix(30, (1, 0, 0)))

    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    #show_image(body_image)
    save_img_hope = im.fromarray(body_image) 
    save_img_hope.save('img_hope2.jpg')

vis_body_transformed(fId=0)
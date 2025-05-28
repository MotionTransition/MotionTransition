

import numpy as np
import torch
from data_loaders.humanml.common.skeleton import Skeleton
from data_loaders.humanml.common.quaternion import *
from data_loaders.humanml.utils.paramUtil import *
np.set_printoptions(suppress=True)

data = np.load('dataset/PerMo/new_joint_vecs_abs_3d/Angry_Hop_A01_001.npy')
device = "cuda:0"

print(data.shape)

j = 0

# root joint rotate, x & z displacement, height
x_global_st = 0
x_global_ed = 4

x_pos_st = x_global_ed
x_pos_ed = x_global_ed + 21 * 3

x_rot_st = x_pos_ed
x_rot_ed = x_pos_ed + 21 * 6

x_vec_st = x_rot_ed
x_vec_ed = x_rot_ed + 22 * 3

l1_loss = 0

for i in range(data.shape[0]):
    # data construct
    root_pos = data[i][ x_global_st + 1 : x_global_ed ]
    root_pos[1], root_pos[2] = root_pos[2], root_pos[1] # x z h -> x h z
    root_pos_copy = root_pos.copy()

    # root 6d rotation
    identity_cont6d = np.array([1,0,0, 0,1,0]) 

    # other joints' 6d rotation
    cont6d_params = np.concatenate(
        [[identity_cont6d], np.array(data[i][x_rot_st:x_rot_ed]).reshape(-1, 6)], axis=0
    )

    # other joints' position
    skel_joints = np.array(data[i][x_pos_st:x_pos_ed]).reshape(-1, 3) # joints pos
    skel_joints = np.concatenate(([root_pos], skel_joints), axis=0) # add root pos to joints

    # tile to bathc size
    skel_joints = np.tile(skel_joints, (64, 1, 1)) # 模拟 batch size
    root_pos = np.tile(root_pos, (64, 1))
    cont6d_params = np.tile(cont6d_params, (64, 1, 1))

    # numpy to tensor
    skel_joints = torch.from_numpy(skel_joints).float().to(device)
    root_pos = torch.from_numpy(root_pos).float().to(device)
    cont6d_params = torch.from_numpy(cont6d_params).float().to(device)
    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)

    # forward kinematics
    skeleton = Skeleton(n_raw_offsets, t2m_kinematic_chain, device)
    Predict = skeleton.forward_kinematics_cont6d(cont6d_params, root_pos, skel_joints=skel_joints)
    Predict = Predict.mean(axis=0)

    Ground_Truth = np.array(data[i][x_pos_st:x_pos_ed]).reshape(-1, 3)
    Ground_Truth = np.concatenate(([root_pos_copy], Ground_Truth))
    Ground_Truth = torch.from_numpy(Ground_Truth).float().to(device)

    l1_loss = l1_loss + torch.abs(Ground_Truth - Predict)
    print((l1_loss).to('cpu').numpy())



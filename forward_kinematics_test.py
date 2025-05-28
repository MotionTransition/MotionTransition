

import numpy as np
import torch
from data_loaders.humanml.common.skeleton import Skeleton
from data_loaders.humanml.common.quaternion import *
from data_loaders.humanml.utils.paramUtil import *
np.set_printoptions(suppress=True)

data = np.load('dataset/PerMo/new_joint_vecs_abs_3d/Angry_Hop_A01_001.npy')

print(data.shape)

i = 0
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

root_pos = data[i][ x_global_st + 1 : x_global_ed ]
root_pos[1], root_pos[2] = root_pos[2], root_pos[1] # x z h -> x h z
root_pos_copy = root_pos.copy()

identity_cont6d = np.array([1,0,0, 0,1,0]) 

cont6d_params = np.concatenate(
    [[identity_cont6d], np.array(data[i][x_rot_st:x_rot_ed]).reshape(-1, 6)], axis=0
)
cont6d_params = np.tile(cont6d_params, (64, 1, 1))
cont6d_params = torch.from_numpy(cont6d_params).float()


skel_joints = np.array(data[i][x_pos_st:x_pos_ed]).reshape(-1, 3) # joints pos
# skel_joints = skel_joints + root_pos # relative root pos to absolute pos
skel_joints = np.concatenate(([root_pos], skel_joints), axis=0) # add root pos to joints

root_pos = np.tile(root_pos, (64, 1))
skel_joints = np.tile(skel_joints, (64, 1, 1)) # 模拟 batch size

skel_joints = torch.from_numpy(skel_joints).float()
root_pos = torch.from_numpy(root_pos).float()


n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
kinematic_chain = t2m_kinematic_chain

skeleton = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
predict_joints_pos = skeleton.forward_kinematics_cont6d(cont6d_params, root_pos, skel_joints=skel_joints)[0].numpy()

# print(f'predict the positions : {predict_joints_pos}')

n_joint_ground_truth = np.array(data[i][x_pos_st:x_pos_ed]).reshape(-1, 3)

# n_joint_ground_truth = n_joint_ground_truth + root_pos_copy

n_joint_ground_truth = np.concatenate(([root_pos_copy], n_joint_ground_truth))

# print(f'position of root joint: {root_pos_copy}')
# print(f'ground-truth of the positions: {n_joint_ground_truth}')

print(root_pos_copy)

print(f'ground-truth, predict')

for i in range(len(n_joint_ground_truth)):
    print(f'{i+1}th joint: {n_joint_ground_truth[i]} vs {predict_joints_pos[i]}')

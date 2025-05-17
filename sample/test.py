# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import DatasetConfig, get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from data_loaders.tensors import collate


import os
import sys
import json
from pprint import pprint
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import DatasetConfig, get_dataset_loader
from utils.model_util import create_model_and_diffusion
from configs import card
import wandb
from data_loaders.humanml.utils.plotting import plot_conditional_samples


from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import cond_synt_args
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader, DatasetConfig
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders import humanml_utils
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from pathlib import Path
from utils.editing_util import get_keyframes_mask, load_fixed_dataset
from data_loaders.humanml.utils.plotting import plot_conditional_samples
import json
from os.path import join as pjoin
from model.style_encoder import StyleClassification
# from data_loaders.amass.utils.utils import batch_to_dict, dict_to_batch


def get_max_length(dataset):
    if dataset in ['kit', 'humanml']:
        return 196
    elif dataset == 'amass':
        return 128
    else:
        return 60

def get_fps(dataset):
    if dataset == 'kit':
        return 12.5
    elif dataset == 'amass':
        return 30
    else:
        return 20

def main():
    args = train_args(base_cls=card.motion_abs_unet_adagn_style100) # Choose the default full motion model from GMD
    args.save_dir = os.path.join("save/test")
    pprint(args.__dict__)

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating data loader...")
    data_conf = DatasetConfig(
        name=args.dataset,
        batch_size=1,
        num_frames=args.num_frames,
        use_abs3d=args.abs_3d,
        traject_only=args.traj_only,
        use_random_projection=args.use_random_proj,
        random_projection_scale=args.random_proj_scale,
        augment_type=args.augment_type,
        std_scale_shift=args.std_scale_shift,
        drop_redundant=args.drop_redundant,
    )

    data = get_dataset_loader(data_conf)

    iterator = iter(data)

    input_motions, model_kwargs = next(iterator)

    input_motions = input_motions.to(dist_util.dev()) # [nsamples, njoints=263/1, nfeats=1/3, nframes=196/200]
    input_masks = model_kwargs["y"]["mask"]  # [nsamples, 1, 1, nframes]
    input_lengths = model_kwargs["y"]["lengths"]  # [nsamples]

    
    args.edit_mode = "benchmark_sparse"
    args.editable_features = "pos_rot_vel"
    args.transition_length = 50
    args.n_keyframes = 5

    model_kwargs['obs_x0'] = input_motions
    model_kwargs['obs_mask'], obs_joint_mask = get_keyframes_mask(data=input_motions, lengths=input_lengths, edit_mode=args.edit_mode,
                                                                  feature_mode=args.editable_features, trans_length=args.transition_length,
                                                                  get_joint_mask=True, n_keyframes=args.n_keyframes) # [nsamples, njoints, nfeats, nframes]


    all_motions = []
    all_lengths = []
    all_text = []
    all_observed_motions = []
    all_observed_masks = []

    args.abs_3d = True

    n_joints = 22
    input_motions = input_motions.cpu().permute(0, 2, 3, 1)
    input_motions = data.dataset.t2m_dataset.inv_transform(input_motions).float()
    input_motions = recover_from_ric(data=input_motions, joints_num=n_joints, abs_3d=args.abs_3d)
    input_motions = input_motions.view(-1, *input_motions.shape[2:]).permute(0, 2, 3, 1)
    all_motions.append(input_motions.cpu().numpy())
    all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())
    all_text += "no_cond"


    input_motions = input_motions.cpu().numpy()
    inpainting_mask = obs_joint_mask.cpu().numpy()



    all_motions = np.stack(all_motions)
    all_lengths = np.stack(all_lengths)
    all_text = np.stack("test")
    all_observed_motions = input_motions # [num_samples, 22, 3, n_frames]
    all_observed_masks = inpainting_mask

    out_path = "./save/test"


    args.num_samples = 1
    args.num_repetitions = 1

    plot_conditional_samples(motion=all_motions,
                             lengths=all_lengths,
                             texts=all_text,
                             observed_motion=all_observed_motions,
                             observed_mask=all_observed_masks,
                             num_samples=args.num_samples,
                             num_repetitions=args.num_repetitions,
                             out_path=out_path,
                             edit_mode=args.edit_mode, #FIXME: only works for selected edit modes
                             stop_imputation_at=0)


def save_multiple_samples(args, out_path, row_print_template, all_print_template, row_file_template, all_file_template,
                          caption, num_samples_in_out_file, rep_files, sample_files, sample_i):
    all_rep_save_file = row_file_template.format(sample_i)
    all_rep_save_path = os.path.join(out_path, all_rep_save_file)
    ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
    hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions}' if args.num_repetitions > 1 else ''
    ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_path}'
    os.system(ffmpeg_rep_cmd)
    print(row_print_template.format(caption, sample_i, all_rep_save_file))
    sample_files.append(all_rep_save_path)
    if (sample_i + 1) % num_samples_in_out_file == 0 or sample_i + 1 == args.num_samples:
        # all_sample_save_file =  f'samples_{(sample_i - len(sample_files) + 1):02d}_to_{sample_i:02d}.mp4'
        all_sample_save_file = all_file_template.format(sample_i - len(sample_files) + 1, sample_i)
        all_sample_save_path = os.path.join(out_path, all_sample_save_file)
        print(all_print_template.format(sample_i - len(sample_files) + 1, sample_i, all_sample_save_file))
        ffmpeg_rep_files = [f' -i {f} ' for f in sample_files]
        vstack_args = f' -filter_complex vstack=inputs={len(sample_files)}' if len(sample_files) > 1 else ''
        ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(
            ffmpeg_rep_files) + f'{vstack_args} {all_sample_save_path}'
        os.system(ffmpeg_rep_cmd)
        sample_files = []
    return sample_files


def construct_template_variables(unconstrained):
    row_file_template = 'sample{:02d}.mp4'
    all_file_template = 'samples_{:02d}_to_{:02d}.mp4'
    if unconstrained:
        sample_file_template = 'row{:02d}_col{:02d}.mp4'
        sample_print_template = '[{} row #{:02d} column #{:02d} | -> {}]'
        row_file_template = row_file_template.replace('sample', 'row')
        row_print_template = '[{} row #{:02d} | all columns | -> {}]'
        all_file_template = all_file_template.replace('samples', 'rows')
        all_print_template = '[rows {:02d} to {:02d} | -> {}]'
    else:
        sample_file_template = 'sample{:02d}_rep{:02d}.mp4'
        sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
        row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
        all_print_template = '[samples {:02d} to {:02d} | all repetitions | -> {}]'

    return sample_print_template, row_print_template, all_print_template, \
           sample_file_template, row_file_template, all_file_template


def load_dataset(args, max_frames, n_frames):
    conf = DatasetConfig(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=max_frames,
        split='test',
        hml_mode='text_only', # 'train'
        use_abs3d=args.abs_3d,
        traject_only=args.traj_only,
        use_random_projection=args.use_random_proj,
        random_projection_scale=args.random_proj_scale,
        augment_type='none',
        std_scale_shift=args.std_scale_shift,
        drop_redundant=args.drop_redundant,
    )
    data = get_dataset_loader(conf)
    data.fixed_length = n_frames
    return data


if __name__ == "__main__":
    main()
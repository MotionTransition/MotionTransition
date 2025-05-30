# This code is based on https://github.com/openai/guided-diffusion,
# and is used to train a diffusion model on human motion sequences.

import sys
sys.path.append('/root/autodl-tmp')

import os
import sys
import json
from pprint import pprint
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import DatasetConfig, get_dataset_loader
from utils.model_util import create_model_and_diffusion, load_saved_model
from configs import card
import wandb


def init_wandb(config, project_name=None, entity=None, tags=[], notes=None, **kwargs):
    # if entity is None:
    #     assert (
    #         "WANDB_ENTITY" in os.environ
    #     ), "Please either pass in \"entity\" to logging.init or set environment variable 'WANDB_ENTITY' to your wandb entity name."
    # if project_name is None:
    #     assert (
    #         "WANDB_PROJECT" in os.environ
    #     ), "Please either pass in \"project_name\" to logging.init or set environment variable 'WANDB_PROJECT' to your wandb project name."
    tags.append(os.path.basename(sys.argv[0]))
    if "_MY_JOB_ID" in os.environ:
        x = f"(jobid:{os.environ['_MY_JOB_ID']})"
        notes = x if notes is None else notes + " " + x
    if len(config.resume_checkpoint) > 0:
        # FIXME: this is specific to the current project's setting
        run_id = config.resume_checkpoint.split("/")[-2]
        wandb.init(project=project_name, entity=entity, config=config, tags=tags, notes=notes, resume="allow", id=run_id, **kwargs)
    else:
        wandb.init(project=project_name, entity=entity, config=config, tags=tags, notes=notes, **kwargs)


def main():
    args = train_args(base_cls=card.motion_abs_unet_adagn_PerMo)
    init_wandb(config=args)
    args.save_dir = os.path.join("save", wandb.run.id)
    pprint(args.__dict__)
    fixseed(args.seed)

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating data loader...")
    if args.no_cond: # 区别在是否加最后两行
        data_conf = DatasetConfig(
            name=args.dataset,
            batch_size=args.batch_size,
            num_frames=args.num_frames,
            use_abs3d=args.abs_3d,
            traject_only=args.traj_only,
            use_random_projection=args.use_random_proj,
            random_projection_scale=args.random_proj_scale,
            augment_type=args.augment_type,
            std_scale_shift=args.std_scale_shift,
            drop_redundant=args.drop_redundant,
            only_text=args.only_text, # 不用reference style的时候加上
            hml_mode="text_only" # 不用reference style的时候加上
        )
    else:
        data_conf = DatasetConfig(
            name=args.dataset,
            batch_size=args.batch_size,
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

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)
    if args.model_path.endswith('.pt'):
        load_saved_model(model, args.model_path)
    model.to(dist_util.dev())
    model.rot2xyz.smpl_model.eval()

    if args.only_text == False:
        # 使用reference style作为条件时，加载风格编码器，并把参数冻结
        load_saved_model(model.style_encoder, args.styenc_dir, style_encoder=True)
        # 冻结 style_encoder
        for param in model.style_encoder.parameters():
            param.requires_grad = False
        model.style_encoder.eval()

    print('Total params: %.2fM' %
          (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(args, model, diffusion, data).run_loop()
    wandb.finish()


if __name__ == "__main__":
    main()


# 模型训练说明

## 调试

在`/.vscode/launch.json`中配置相关条目

## 控制台训练

### 1. 前置条件 Visual Studio Code 添加 Python 运行时环境

```shell
export PYTHONPATH="${PWD}:${PYTHONPATH}"
# or
export PYTHONPATH="/root/autodl-tmp/MotionTransition3/MotionTransition/:${PWD}:${PYTHONPATH}"
```

### 2. 训练无风格情况下的模型

基于之前的模型进行训练

```sh
# 前台运行
python train/train_condmdi.py --model_path ./save/model000550000.pt --save_interval 10000 --keyframe_conditioned --only_text
# 后台运行
nohup python train/train_condmdi.py --model_path ./save/model000550000.pt --save_interval 10000 --keyframe_conditioned --only_text &
```

不基于之前的模型训练

```sh
# 后台运行
nohup python train/train_condmdi.py --num_steps 100000  --save_interval 5000 --keyframe_conditioned --only_text &
```

在 nohup.txt 中查看 wandb 链接，并在 wandb 中结束程序运行。

修改 loss 值：`diffusion/gaussian_diffusion.py`


### 3. 带风格编码器的训练

```sh
python ./train/train_condmdi.py --keyframe_conditioned --styenc_dir ./pretrained_model/style_encoder_500_000.pt --save_interval 10000
# 后台运行
nohup python ./train/train_condmdi.py --keyframe_conditioned --styenc_dir ./pretrained_model/style_encoder_500_000.pt --save_interval 10000 &
```
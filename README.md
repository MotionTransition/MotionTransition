

添加Python运行时环境

```shell
export PYTHONPATH="${PWD}:${PYTHONPATH}"

export PYTHONPATH="/root/autodl-tmp/MotionTransition3/MotionTransition/:${PWD}:${PYTHONPATH}"
```

训练无风格情况下的模型

```sh
python train/train_condmdi.py --model_path ./pretrain/model000550000.pt --save_interval 10000 --keyframe_conditioned --only_text

nohup python train/train_condmdi.py --model_path ./pretrain/model000550000.pt --save_interval 10000 --keyframe_conditioned --only_text &
```
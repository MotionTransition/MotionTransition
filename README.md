# MotionTransition V1

A(动作生成)+B(风格融合)工作，基于CondMDI源码

## 简单说明

- 训练，见`/train/README.md`
    - `/train/train_condmdi.py` 文件训练模型
    - `/train/train_condmdi_style.py` 文件训练带风格编码器的模型
- `/sample/style_synthesis.py` 文件使用训练好的模型进行预测

## 文件结构

```bash
.
├── body_models // Upload Separately, unzip body_models.zip
    ├── smpl
    └── smpl.zip
├── configs
├── data_loaders
├── dataset // Upload Separately, unzip dataset.zip
    ├── 000021.npy
    ├── cp_dataset.py
    ├── HumanML3D_abs
    ├── humanml_opt.txt
    ├── inv_rand_proj.npy
    ├── kit_mean.npy
    ├── kit_opt.txt
    ├── kit_std.npy
    ├── PerMo
    ├── permo_opt.txt
    ├── rand_proj.npy
    ├── README.md
    ├── style100_opt.txt
    ├── t2m_mean.npy
    └── t2m_std.npy
├── diffusion
├── eval
├── glove // Upload Separately, unzip glove.zip
    ├── our_vab_data.npy
    ├── our_vab_idx.pkl
    └── our_vab_words.pkl
├── mld
├── model
├── prepare
├── pretrained_model // Upload Separately, unzip pretrained_model.zip
    ├── style_encoder_200_000.pt
    └── style_encoder_500_000.pt
├── sample
├── save // 模型保存位置
├── t2m // Upload Separately, unzip t2m.zip
    ├── kit
    │   ├── Comp_v6_KLD005
    │   │   ├── meta
    │   │   └── model
    │   ├── text_mot_match
    │   │   ├── eval
    │   │   └── model
    │   └── VQVAEV3_CB1024_CMT_H1024_NRES3
    │       ├── meta
    │       └── model
    └── t2m
        ├── ._.DS_Store
        ├── ._kit
        ├── ._t2m
        ├── Comp_v6_KLD005
        │   └── meta
        ├── Comp_v6_KLD01
        │   ├── meta
        │   └── model
        ├── text_mot_match
        │   ├── eval
        │   └── model
        └── VQVAEV3_CB1024_CMT_H1024_NRES3
            ├── meta
            └── model
├── train
├── utils
└── wandb // wandb 运行日志
```
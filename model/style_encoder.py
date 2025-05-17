import torch
import torch.nn as nn
from torch import  nn
from mld.models.architectures.tools.embeddings import (TimestepEmbedding,
                                                       Timesteps)
from mld.models.operator import PositionalEncoding
from mld.models.operator.cross_attention import (SkipTransformerEncoder,
                                                 TransformerEncoder,
                                                 TransformerEncoderLayer)
from mld.models.operator.position_encoding import build_position_encoding
from mld.utils.temos_utils import lengths_to_mask
import torch.nn.functional as F
from torch import Tensor, nn
from typing import List, Optional, Union
from torch.distributions.distribution import Distribution
from torch.optim import AdamW
from torch.nn import Parameter
import math

# class FocalLoss(nn.Module):

#     def __init__(self, gamma=0, eps=1e-7):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.eps = eps
#         self.ce = torch.nn.CrossEntropyLoss()

#     def forward(self, input, target):
#         logp = self.ce(input, target)
#         p = torch.exp(-logp)
#         loss = (1 - p) ** self.gamma * logp
#         return loss.mean()
    
def conv_layer(kernel_size, in_channels, out_channels, pad_type='replicate'):
    def zero_pad_1d(sizes):
        return nn.ConstantPad1d(sizes, 0)

    if pad_type == 'reflect':
        pad = nn.ReflectionPad1d
    elif pad_type == 'replicate':
        pad = nn.ReplicationPad1d
    elif pad_type == 'zero':
        pad = zero_pad_1d

    pad_l = (kernel_size - 1) // 2
    pad_r = kernel_size - 1 - pad_l
    return nn.Sequential(pad((pad_l, pad_r)), nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size))

class StyleClassification(nn.Module):
    def __init__(self,
                 latent_dim: list = [1, 512],
                 ff_size: int = 1024,
                 num_layers: int = 6,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 position_embedding: str = "learned",
                 **kwargs) -> None:
        
        super().__init__()
        self.latent_dim = latent_dim[-1]
        self.skel_embedding = nn.Linear(263, self.latent_dim)
        self.latent_size = latent_dim[0]
        self.global_motion_token = nn.Parameter(
                torch.randn(self.latent_size * 2, self.latent_dim))

        self.abl_plus = False
        self.cond_mode = 'style'

        self.query_pos = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding)
        self.mem_pos = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding)

        encoder_layer_s = TransformerEncoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )

        encoder_norm = nn.LayerNorm(self.latent_dim)
        self.encoder = TransformerEncoder(encoder_layer_s, num_layers,encoder_norm)
        self.triplet_loss = nn.TripletMarginLoss(margin=5, reduction='none')
    
    def forward(
            self,
            features: Tensor,
            lengths: Optional[List[int]] = None,
            skip = False,
    ) -> Union[Tensor, Distribution]:
        if lengths is None:
            lengths = [feature.shape[-1] for feature in features]

        device = features.device

        bs, nfeats, njoints, nframes = features.shape
        # max_length = max(lengths)
        max_length = nframes
        # if bs == 1:
        #     max_length = nframes
        # max_length = 224
        mask = lengths_to_mask(lengths, device, max_length)

        x = features.float()
        # Embed each human poses into latent vectors
        
        x = x.permute(0, 3, 2, 1)

        # if torch.isnan(x).any():
        #     print("x has NaN!")

        if skip == False:
            x = self.skel_embedding(x)
        
        x = x.squeeze(2)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        # Each batch has its own set of tokens
        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))

        # create a bigger mask, to allow attend to emb
        dist_masks = torch.ones((bs, dist.shape[0]),
                                dtype=bool,
                                device=x.device)
        aug_mask = torch.cat((dist_masks, mask), 1)

        # adding the embedding token for all sequences

        xseq = torch.cat((dist, x), 0)
        xseq = self.query_pos(xseq)

        dist = self.encoder(xseq,src_key_padding_mask=~aug_mask)

        return dist[0]
    
    def parameters_wo_clip(self):
        return [
            p for name, p in self.named_parameters()
        ]
    
    def training_losses(self,
                        model,
                        x_start,
                        t,
                        model_kwargs=None,
                        noise=None,
                        dataset=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        # [bs, 4 or 263, 1, seqlen]
        bs, njoints, nfeats, nframes = x_start.shape
        model_kwargs['y']['x_start'] = x_start
        y = model_kwargs['y']

        motion_t_lengths = []
        if 'motion_t' in y:
            motion_t_list = []
            for t in y['motion_t']:
                t = t.float().unsqueeze(1).permute(2, 1, 0)
                current_len = t.shape[-1]
                motion_t_lengths.append(current_len)
                if current_len >= nframes:
                    motion_t_list.append(t[:nframes])
                else:
                    pad_size = nframes - current_len
                    processed_tensor = F.pad(t, (0, pad_size), value=0.0)
                    motion_t_list.append(processed_tensor)
            motion_t = (torch.stack(motion_t_list, dim=0)).to("cuda:0")
            motion_t_lengths_torch = (torch.tensor(motion_t_lengths)).to("cuda:0")
        motion_s_lengths = []
        if 'motion_s' in y:
            motion_s_list = []
            for t in y['motion_s']:
                t = t.float().unsqueeze(1).permute(2, 1, 0)
                current_len = t.shape[-1]
                motion_s_lengths.append(current_len)
                if current_len >= nframes:
                    motion_s_list.append(t[:nframes])
                else:
                    pad_size = nframes - current_len
                    processed_tensor = F.pad(t, (0, pad_size), value=0.0)
                    motion_s_list.append(processed_tensor)
            motion_s = (torch.stack(motion_s_list, dim=0)).to("cuda:0")
            motion_s_lengths_torch = (torch.tensor(motion_s_lengths)).to("cuda:0")

        if model_kwargs is None:
            model_kwargs = {}

        terms = {}

        enc_style_self = self.forward(x_start, y['lengths'])
        enc_style_s = self.forward(motion_s, motion_s_lengths_torch)
        enc_style_t = self.forward(motion_t, motion_t_lengths_torch)

        terms["trip_loss"] = self.triplet_loss(enc_style_self, enc_style_t, enc_style_s)
            
        if torch.isnan(terms["trip_loss"]).any():
            print("trip_loss has nan!")

        terms["loss"] = terms.get('trip_loss', 0.)

        return terms

class StyleEncoder(nn.Module):
    # style_nums是风格的种类，permo有34种风格，如果训练其他数据集，注意调整
    def __init__(self,
                 latent_dim = 512,
                 style_nums = 34) -> None:
        
        super().__init__()

        self.cond_mode = 'style'

        self.encoder = StyleClassification()
        # Classifier
        self.classifier = nn.Linear(latent_dim, style_nums)

        self.criterion = nn.CrossEntropyLoss()
        
        self.triplet_loss = nn.TripletMarginLoss(margin=5, reduction='none')

    def forward(
            self,
            features: Tensor,
            lengths: Optional[List[int]] = None,
            skip = False,
    ) -> Union[Tensor, Distribution]:
        style_code = self.encoder(features=features, lengths=lengths)
        pred = self.classifier(style_code)
        return style_code, pred

    def training_losses(self,
                        model,
                        x_start,
                        t,
                        model_kwargs=None,
                        noise=None,
                        dataset=None):

        # [bs, 4 or 263, 1, seqlen]
        _, _, _, nframes = x_start.shape
        y = model_kwargs['y']

        motion_t_lengths = []
        motion_t_list = []
        for tmp in model_kwargs['y']['motion_t']:
            tmp = tmp.float().unsqueeze(1).permute(2, 1, 0)
            current_len = tmp.shape[-1]
            motion_t_lengths.append(current_len)
            if current_len >= nframes:
                motion_t_list.append(tmp[:, :, :nframes])
            else:
                pad_size = nframes - current_len
                processed_tensor = F.pad(tmp, (0, pad_size), value=0.0)
                motion_t_list.append(processed_tensor)
        motion_t = (torch.stack(motion_t_list, dim=0)).to("cuda:0")
        motion_t_lengths_torch = (torch.tensor(motion_t_lengths)).to("cuda:0")
        
        motion_s_lengths = []
        motion_s_list = []
        for tmp in model_kwargs['y']['motion_s']:
            tmp = tmp.float().unsqueeze(1).permute(2, 1, 0)
            current_len = tmp.shape[-1]
            motion_s_lengths.append(current_len)
            if current_len >= nframes:
                motion_s_list.append(tmp[:, :, :nframes])
            else:
                pad_size = nframes - current_len
                processed_tensor = F.pad(tmp, (0, pad_size), value=0.0)
                motion_s_list.append(processed_tensor)
        motion_s = (torch.stack(motion_s_list, dim=0)).to("cuda:0")
        motion_s_lengths_torch = (torch.tensor(motion_s_lengths)).to("cuda:0")

        style_code, pred = self.forward(x_start, y['lengths'])
        style_code_t, _ = self.forward(motion_t, motion_t_lengths_torch)
        style_code_s, _ = self.forward(motion_s, motion_s_lengths_torch)
        batch_style_ids = model_kwargs['y']['style_ids']
        # 输出到txt文件，在检查预训练的风格编码器效果时，取消注释
        # output_path = "pretrain_style_encoder.txt"

        # with open(output_path, "a") as f:
        #     for name, feature in zip(model_kwargs['y']['style_name'], style_code):
        #         # 将张量转换为字符串格式（例如保留4位小数）
        #         feature_str = " ".join([f"{x:.4f}" for x in feature.tolist()])
        #         # 写入名称 + 特征向量
        #         f.write(f"{name} {feature_str}\n")

        terms = {}
        terms['loss'] = self.criterion(pred, batch_style_ids) + self.triplet_loss(style_code, style_code_t, style_code_s)
        # terms['loss'] = self.criterion(pred, batch_style_ids)

        return terms
    
    def parameters_wo_clip(self):
        return [
            p for name, p in self.named_parameters()
        ]
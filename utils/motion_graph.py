import torch
import numpy as np

def blend_motion(A: torch.Tensor, B: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    混合两个动作序列（考虑每个样本的真实长度）
    首尾帧完全为A，中间帧完全为B（抛物线过渡）
    
    参数:
        A: 张量，形状为 (batch, features, 1, num_frames)
        B: 张量，形状与A相同
        lengths: 张量，形状为 (batch,)，记录每个样本的真实帧数
    
    返回:
        C: 混合后的张量，形状与A相同
    """
    assert A.shape == B.shape, "A和B的形状必须相同"
    batch_size, features, _, num_frames = A.shape
    assert lengths.shape == (batch_size,), "lengths的形状必须为(batch,)"
    assert torch.all(lengths <= num_frames), "存在lengths超过num_frames的情况"
    
    device = A.device
    C = A.clone()  # 初始化结果为A的副本
    
    for b in range(batch_size):
        L = lengths[b].item()
        if L <= 1:
            continue  # 不需要混合的情况
            
        # 生成时间轴（只考虑有效长度）
        t = torch.arange(L, dtype=torch.float32, device=device)
        T = L - 1  # 最后一帧的索引
        middle_frame = L // 2  # 自动向下取整
        
        # -------------------------------
        # 构造抛物线系数（仅在有效长度内）
        # -------------------------------
        if middle_frame == 0 or middle_frame == T:
            # 极短序列的特殊处理
            alpha = torch.zeros(L, device=device)
            alpha[middle_frame] = 1.0
        else:
            # 解方程：α(t) = a*t² + b*t
            # 约束条件：α(0)=0, α(middle)=1, α(T)=0
            denominator = middle_frame**2 - T * middle_frame
            a = -1 / denominator
            b_coeff = -a * T
            alpha = a * t**2 + b_coeff * t
            
            # 数值稳定性处理
            alpha = torch.clamp(alpha, 0, 1)
        
        # -------------------------------
        # 应用混合（仅修改有效长度部分）
        # -------------------------------
        # 调整alpha形状为 (1, features, 1, L)
        alpha = alpha.view(1, 1, 1, L)
        
        # 混合公式：C = (1-α)*A + α*B
        C[b, :, :, :L] = (1 - alpha) * A[b, :, :, :L] + alpha * B[b, :, :, :L]
    
    return C

def dual_blend_motion(A: torch.Tensor, B: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    双向混合动作序列
    - 前1/4长度：A→B融合 (alpha从0到1)
    - 中间1/2长度：保持B
    - 后1/4长度：B→A融合 (alpha从1到0)
    
    参数:
        A, B: 形状 (batch, features, 1, max_frames)
        lengths: 每个样本的实际长度 (batch,)
    """
    assert A.shape == B.shape, "A和B形状必须相同"
    batch_size, features, _, max_frames = A.shape
    device = A.device
    
    C = A.clone()  # 初始化结果为A的副本
    
    for b in range(batch_size):
        L = lengths[b].item()
        if L <= 1:
            continue
            
        # 划分三个阶段
        transition_quarter = L // 4
        phase1_end = transition_quarter          # A→B过渡结束位置
        phase3_start = L - transition_quarter    # B→A过渡开始位置
        
        # 生成时间轴
        t = torch.arange(L, dtype=torch.float32, device=device)
        
        # -------------------------------
        # 构造三段式alpha系数
        # -------------------------------
        alpha = torch.zeros(L, device=device)
        
        # Phase 1: A→B过渡 (前1/4，抛物线从0到1)
        if phase1_end > 0:
            x = t[:phase1_end] / (phase1_end - 1)  # 归一化到[0,1]
            alpha[:phase1_end] = 2*x - x**2  # 二次函数加速上升
        
        # Phase 2: 保持B (中间1/2)
        if phase3_start > phase1_end:
            alpha[phase1_end:phase3_start] = 1.0
        
        # Phase 3: B→A过渡 (后1/4，抛物线从1到0)
        if phase3_start < L:
            x = (t[phase3_start:] - phase3_start) / (L - phase3_start - 1)  # 归一化到[0,1]
            alpha[phase3_start:] = 1 - (2*x - x**2)  # 二次函数减速下降
        
        # -------------------------------
        # 应用混合 (形状适配)
        # -------------------------------
        alpha = alpha.view(1, 1, 1, L)
        C[b, :, :, :L] = (1 - alpha) * A[b, :, :, :L] + alpha * B[b, :, :, :L]
    
    return C

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

class IndexBasedInterpolator:
    def __init__(self, joint_num=22):
        # 定义各部分参数的索引范围
        self.index_config = {
            'root_rot_vel': (0, 1),         # 索引0
            'root_lin_vel': (1, 3),         # 索引1-2
            'root_y': (3, 4),               # 索引3
            'ric_data': (4, 4 + (joint_num-1)*3),      # 索引4-66 (21*3=63)
            'rot_data': (67, 67 + (joint_num-1)*6),    # 索引67-192 (21*6=126)
            'local_velocity': (193, 193 + joint_num*3),# 索引193-258 (22*3=66)
            'foot_contact': (259, 263)      # 索引259-262 (4)
        }
        self.joint_num = joint_num

    def split_features(self, frame):
        """将263维向量拆分为各运动参数"""
        params = {}
        # 根节点参数
        params['root_rot_vel'] = frame[..., self.index_config['root_rot_vel'][0]:self.index_config['root_rot_vel'][1]]
        params['root_lin_vel'] = frame[..., self.index_config['root_lin_vel'][0]:self.index_config['root_lin_vel'][1]]
        params['root_y'] = frame[..., self.index_config['root_y'][0]:self.index_config['root_y'][1]]
        
        # 关节参数 
        ric_start, ric_end = self.index_config['ric_data']
        params['ric_data'] = frame[..., ric_start:ric_end].reshape(-1, self.joint_num-1, 3)
        
        rot_start, rot_end = self.index_config['rot_data']
        params['rot_data'] = frame[..., rot_start:rot_end].reshape(-1, self.joint_num-1, 6)
        
        # 其他参数
        vel_start, vel_end = self.index_config['local_velocity']
        params['local_velocity'] = frame[..., vel_start:vel_end].reshape(-1, self.joint_num, 3)
        
        fc_start, fc_end = self.index_config['foot_contact']
        params['foot_contact'] = frame[..., fc_start:fc_end]
        return params

    def interpolate(self, frame1, frame2, alpha=0.5):
        """主插值函数"""
        # 分割参数
        p1 = self.split_features(frame1)
        p2 = self.split_features(frame2)
        
        # 各分量插值
        interp_params = {}
        
        # ---- 根运动插值 ----
        interp_params['root_rot_vel'] = self._slerp_rotvec(p1['root_rot_vel'], p2['root_rot_vel'], alpha)
        interp_params['root_lin_vel'] = self._lerp(p1['root_lin_vel'], p2['root_lin_vel'], alpha)
        interp_params['root_y'] = self._quadratic_interp(p1['root_y'], p2['root_y'], alpha)
        
        # ---- 关节位置插值 ----
        interp_ric = self._lerp(p1['ric_data'], p2['ric_data'], alpha)
        interp_params['ric_data'] = interp_ric.reshape(-1, (self.joint_num-1)*3)
        
        # ---- 关节旋转插值 ----
        interp_rot = []
        for j in range(self.joint_num-1):
            rot6d_1 = p1['rot_data'][:, j]
            rot6d_2 = p2['rot_data'][:, j]
            interp_rot.append(self._slerp_6d(rot6d_1, rot6d_2, alpha))
        interp_params['rot_data'] = torch.stack(interp_rot, dim=1).reshape(-1, (self.joint_num-1)*6)
        
        # ---- 局部速度插值 ----
        interp_vel = self._lerp(p1['local_velocity'], p2['local_velocity'], alpha)
        interp_params['local_velocity'] = interp_vel.reshape(-1, self.joint_num*3)
        
        # ---- 脚部接触状态 ----
        interp_params['foot_contact'] = self._process_foot_contact(p1['foot_contact'], p2['foot_contact'], alpha)
        
        # 合并为263维向量
        return self.combine_features(interp_params)

    def combine_features(self, params):
        """将各参数合并回263维向量"""
        components = [
            params['root_rot_vel'],
            params['root_lin_vel'],
            params['root_y'],
            params['ric_data'],
            params['rot_data'],
            params['local_velocity'],
            params['foot_contact']
        ]
        return torch.cat(components, dim=-1)

    # ---------- 插值核心方法 ----------
    def _slerp_rotvec(self, v1, v2, alpha):
        """旋转向量球面插值"""
        rot1 = R.from_rotvec(v1.cpu().numpy())
        rot2 = R.from_rotvec(v2.cpu().numpy())
        interp_rot = R.slerp(rot1, rot2, alpha)
        return torch.tensor(interp_rot.as_rotvec(), dtype=torch.float32).to(v1.device)

    def _slerp_6d(self, rot6d_1, rot6d_2, alpha):
        """6D旋转表示插值"""
        # 转换为旋转矩阵
        rotmat1 = self._6d_to_rotmat(rot6d_1)
        rotmat2 = self._6d_to_rotmat(rot6d_2)
        
        # 插值旋转矩阵
        interp_rotmat = torch.matrix_exp(
            alpha * torch.matrix_log(torch.bmm(rotmat2, rotmat1.transpose(1, 2)))
        ).bmm(rotmat1)
        
        # 转回6D表示
        return interp_rotmat[:, :, :2].reshape(-1, 6)

    def _6d_to_rotmat(self, rot6d):
        """6D转旋转矩阵"""
        a1, a2 = rot6d[..., :3], rot6d[..., 3:]
        b1 = torch.nn.functional.normalize(a1, dim=-1)
        b2 = a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1
        b2 = torch.nn.functional.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-2)

    def _lerp(self, x, y, alpha):
        """线性插值"""
        return (1 - alpha) * x + alpha * y

    def _quadratic_interp(self, y1, y2, alpha):
        """根高度的抛物线插值（防止穿模）"""
        mid = (y1 + y2) / 2 + 0.1 * (1 - (2*alpha-1)**2)  # 0.1为重力系数
        return (1 - alpha) * y1 + alpha * y2 + 4 * alpha * (1 - alpha) * (mid - (y1 + y2)/2)

    def _process_foot_contact(self, fc1, fc2, alpha):
        """脚部接触状态插值策略"""
        # alpha在中间30%区域保持接触状态
        threshold = 0.3
        if alpha < 0.5 - threshold/2: 
            return fc1
        elif alpha > 0.5 + threshold/2:
            return fc2
        else:
            return torch.maximum(fc1, fc2)
        
    def interpolate_between(self, frame1, frame2, num_frames=5, include_ends=False):
        """
        输入：
        - frame1: 起始帧 (形状 [B, 263])
        - frame2: 结束帧 (形状 [B, 263])
        - num_frames: 需要生成的中间帧数量
        - include_ends: 是否包含起始/结束帧在输出中
        输出：
        - interp_frames: 插值后的序列 (形状 [num_frames, 263] 或 [num_frames+2, 263])
        """
        # 生成alpha值序列
        if include_ends:
            alphas = torch.linspace(0, 1, num_frames + 2)
        else:
            alphas = torch.linspace(0, 1, num_frames + 2)[1:-1]
        
        # 批量插值计算
        interp_frames = []
        for alpha in alphas:
            interp_frame = self.interpolate(frame1, frame2, alpha.item())
            interp_frames.append(interp_frame)
        
        return torch.cat(interp_frames, dim=0)

def linear_interpolate(start_frame, end_frame, num_frames):
    """
    线性插值函数
    输入：
    - start_frame: 起始帧，形状 (feature_dim,)
    - end_frame: 结束帧，形状 (feature_dim,)
    - num_frames: 插值帧数
    输出：
    - 插值结果，形状 (num_frames, feature_dim)
    """
    # 在特征维度上插值
    alpha = torch.linspace(0, 1, num_frames + 2, device=start_frame.device)[1:-1]  # 去掉首尾
    interpolated = start_frame.unsqueeze(0) * (1 - alpha.unsqueeze(1)) + end_frame.unsqueeze(0) * alpha.unsqueeze(1)
    return interpolated

def dual_motion_transition(A, B, lengths_A, lengths_B, feature_dim=263, 
                          transition_frames=5, dist_p=2):
    """
    改进版本：考虑每个样本的实际长度，避免使用填充帧
    
    输入：
    - A, B: 形状 (batch, 263, 1, max_frames)
    - lengths_A: 形状 (batch,), 表示每个样本A的实际长度
    - lengths_B: 形状 (batch,), 表示每个样本B的实际长度
    """
    assert A.shape[0] == B.shape[0] == len(lengths_A) == len(lengths_B), "Batch不匹配"
    assert transition_frames % 2 == 1, "过渡帧数需为奇数"

    interpolator = IndexBasedInterpolator(joint_num=22)
    # 提取有效特征 (batch, frames, 263)
    features_A = A.permute(0, 3, 1, 2).squeeze(-1)  # (batch, max_A, 263)
    features_B = B.permute(0, 3, 1, 2).squeeze(-1)  # (batch, max_B, 263)

    # 计算距离矩阵时屏蔽填充区域
    dist_A2B = _masked_cdist(features_A, features_B, lengths_A, lengths_B, p=dist_p)
    dist_B2A = _masked_cdist(features_B, features_A, lengths_B, lengths_A, p=dist_p)

    res = []
    for b in range(A.shape[0]):
        single_res = []
        len_A = lengths_A[b].item()
        len_B = lengths_B[b].item()

        # --- 第一次过渡 A->B ---
        # 有效区域限制
        valid_A = np.arange(1, len_A//4-5)
        valid_B = np.arange(len_B//4, len_B//2)
        min_idx = torch.argmin(dist_A2B[b][valid_A][:, valid_B])
        a_idx, b_idx = np.unravel_index(min_idx.cpu().numpy(), (len(valid_A), len(valid_B)))
        a_start = valid_A[a_idx]
        b_start = valid_B[b_idx]

        # --- 第二次过渡 B->A ---
        valid_B_return = np.arange(len_B//2, len_B*3//4)
        valid_A_return = np.arange(len_A*3//4+5, len_A)
        min_idx_return = torch.argmin(dist_B2A[b][valid_B_return][:, valid_A_return])
        b_ret_idx, a_ret_idx = np.unravel_index(
            min_idx_return.cpu().numpy(), (len(valid_B_return), len(valid_A_return))
        )
        b_ret_start = valid_B_return[b_ret_idx]
        a_ret_start = valid_A_return[a_ret_idx]

        # 合成结果
        single_res = [
            features_A[b][:a_start + 1].cpu().numpy(),  # A 的前半部分
            linear_interpolate(features_A[b][a_start], features_B[b][b_start], num_frames=b_start - a_start - 1).cpu().numpy(),  # A->B 过渡
            features_B[b][b_start:b_ret_start + 1].cpu().numpy(),  # B 的中间部分
            linear_interpolate(features_B[b][b_ret_start], features_A[b][a_ret_start], num_frames=a_ret_start - b_ret_start - 1).cpu().numpy(),  # B->A 过渡
            features_A[b][a_ret_start:].cpu().numpy()  # A 的后半部分
        ]

        # 将结果拼接为一个完整的序列
        single_res = np.concatenate(single_res, axis=0)
        res.append(single_res)
    res = torch.tensor(res).to("cuda:0")
    res = res.unsqueeze(0).permute(1, 3, 0, 2)
    return res
        

    # 应用混合（仅使用有效长度内的数据）
    # blended = _apply_safe_blend(A, B, transitions, transition_frames, lengths_A, lengths_B)
    # return blended, transitions

def _masked_cdist(x1, x2, len1, len2, p=2):
    """计算带掩码的距离矩阵，将无效区域设为无穷大"""
    batch, max1, dim = x1.shape
    max2 = x2.shape[1]
    
    # 全量距离矩阵
    dist = torch.cdist(x1, x2, p=p)  # (batch, max1, max2)
    
    # 为每个batch添加掩码
    for b in range(batch):
        l1 = len1[b].item()
        l2 = len2[b].item()
        # 将x1超出l1的部分设为inf
        if l1 < max1:
            dist[b, l1:, :] = float('inf')
        # 将x2超出l2的部分设为inf
        if l2 < max2:
            dist[b, :, l2:] = float('inf')
    return dist

def _get_valid_range(length, half, min_start=0):
    """生成有效帧索引，确保过渡区间不越界"""
    start = max(half, min_start)
    end = length - half
    return np.arange(start, end) if start < end else np.array([])
# -*- coding: utf-8 -*-
"""预测网络

七个飞行状态输入经共享编码器与相位子网络得到状态表示，机型静态特征另走一路，
两路非线性融合后输出 FF1 与 FF2。机型这一路可以关闭，相位这一路有三种模式，
都用于消融。
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from af_data import AIRCRAFT_FEATURE_NAMES, FEATURE_NAMES, PHASE_NAMES

# 相位这一路的三种模式
#   per_phase 六个子网按相位硬路由，完整模型
#   shared    换成一个共用子网，形状、深度、残差不变，只去掉按相位取权重
#   none      整块去掉，共享编码器的输出直接进融合层
PHASE_MODES = ('per_phase', 'shared', 'none')


class PhaseSubnetworks(nn.Module):
    """六个相位子网络 64 → 64 → 32 → 64

    按每个样本的相位标签取出对应的一组权重再做批量矩阵乘，与逐个相位分别前向
    完全等价，只是把六次掩码取样合并成一次算子。
    """

    def __init__(self, hidden_dim: int, num_phases: int):
        super().__init__()
        mid_dim = hidden_dim // 2
        shapes = ((hidden_dim, hidden_dim), (mid_dim, hidden_dim), (hidden_dim, mid_dim))
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.empty(num_phases, out_dim, in_dim))
             for out_dim, in_dim in shapes])
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(num_phases, out_dim)) for out_dim, _ in shapes])
        for weight in self.weights:
            for phase_id in range(num_phases):
                nn.init.xavier_normal_(weight[phase_id])

    def forward(self, h: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        for depth, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            h = torch.baddbmm(bias[phases].unsqueeze(2), weight[phases],
                              h.unsqueeze(2)).squeeze(2)
            if depth < len(self.weights) - 1:
                h = torch.tanh(h)
        return h


class FuelFlowNet(nn.Module):

    def __init__(self,
                 aircraft_feature_table: torch.Tensor,
                 aircraft_feature_mean: torch.Tensor,
                 aircraft_feature_scale: torch.Tensor,
                 input_dim: int = len(FEATURE_NAMES),
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 aircraft_feature_dim: int = 16,
                 use_aircraft_features: bool = True,
                 phase_mode: str = 'per_phase'):
        super().__init__()
        if input_dim != len(FEATURE_NAMES):
            raise ValueError(f'输入维度应为 {len(FEATURE_NAMES)}，收到 {input_dim}')
        if num_layers < 1:
            raise ValueError('num_layers 至少为 1')
        if hidden_dim < 2:
            raise ValueError('hidden_dim 至少为 2')

        table = aircraft_feature_table.detach().clone().float()
        if table.ndim != 2 or table.shape[1] != len(AIRCRAFT_FEATURE_NAMES):
            raise ValueError('机型特征表形状不正确')
        mean = aircraft_feature_mean.detach().clone().float()
        scale = aircraft_feature_scale.detach().clone().float()
        if mean.shape != (table.shape[1],) or scale.shape != (table.shape[1],):
            raise ValueError('机型特征标准化参数维度不正确')
        scale = torch.where(scale < 1e-8, torch.ones_like(scale), scale)

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.aircraft_feature_dim = int(aircraft_feature_dim)
        if phase_mode not in PHASE_MODES:
            raise ValueError(f'phase_mode 应为 {PHASE_MODES} 之一，收到 {phase_mode}')
        self.use_aircraft_features = bool(use_aircraft_features)
        self.phase_mode = str(phase_mode)

        self.register_buffer('aircraft_feature_table', table)
        self.register_buffer('aircraft_feature_mean', mean)
        self.register_buffer('aircraft_feature_scale', scale)
        self.register_buffer('x_mean', torch.zeros(input_dim))
        self.register_buffer('x_scale', torch.ones(input_dim))

        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(self.num_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
            in_dim = hidden_dim
        self.shared_encoder = nn.Sequential(*layers)

        # shared 只建一组权重，所有样本都走它：形状、深度、残差都不变，
        # 去掉的只是"按相位取不同权重"这一步；none 连这一块都没有
        if self.phase_mode == 'none':
            self.phase_networks = None
        else:
            self.phase_networks = PhaseSubnetworks(
                hidden_dim,
                len(PHASE_NAMES) if self.phase_mode == 'per_phase' else 1)

        if self.use_aircraft_features:
            self.aircraft_encoder = nn.Sequential(
                nn.Linear(len(AIRCRAFT_FEATURE_NAMES), self.aircraft_feature_dim),
                nn.Tanh(),
            )
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim + self.aircraft_feature_dim, hidden_dim),
                nn.Tanh(),
            )
        else:
            self.aircraft_encoder = None
            self.fusion = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())

        self.output_layer = nn.Linear(hidden_dim, 2)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def set_input_normalization(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        mean = mean.detach().to(self.x_mean)
        scale = scale.detach().to(self.x_scale)
        if mean.shape != self.x_mean.shape or scale.shape != self.x_scale.shape:
            raise ValueError('飞行状态标准化参数维度不正确')
        self.x_mean.copy_(mean)
        self.x_scale.copy_(torch.where(scale < 1e-8, torch.ones_like(scale), scale))

    def normalized_aircraft_features(self, aircraft_ids: torch.Tensor) -> torch.Tensor:
        # 机型 ID 与相位编号在构造数据集时已按词表映射，越界会在取行时直接报错，
        # 此处不再逐次取最值校验，那会在每次前向上触发一次设备同步
        ids = aircraft_ids.long().view(-1)
        raw = self.aircraft_feature_table[ids]
        return (raw - self.aircraft_feature_mean) / self.aircraft_feature_scale

    def forward(self, x: torch.Tensor, phase: torch.Tensor,
                aircraft_ids: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f'飞行状态输入应为 [N, {self.input_dim}]')
        phases = phase.long().view(-1)
        aircraft = aircraft_ids.long().view(-1)
        if len(x) != len(phases) or len(x) != len(aircraft):
            raise ValueError('飞行状态、相位与机型的样本数不一致')
        shared = self.shared_encoder((x - self.x_mean) / self.x_scale)
        if self.phase_networks is None:
            state = shared
        else:
            routing = (phases if self.phase_mode == 'per_phase'
                       else torch.zeros_like(phases))
            state = shared + self.phase_networks(shared, routing)

        if self.use_aircraft_features:
            aircraft_repr = self.aircraft_encoder(self.normalized_aircraft_features(aircraft))
            fused = self.fusion(torch.cat([state, aircraft_repr], dim=1))
        else:
            fused = self.fusion(state)
        return F.softplus(self.output_layer(fused)) + 1e-6

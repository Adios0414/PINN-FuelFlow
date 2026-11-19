# -*- coding: utf-8 -*-
"""损失

    L = L_data + a·L_mass + b·L_force + c·L_phase

L_data   对 FF1 与 FF2 的均方误差。
L_mass   连续窗口内预测油耗的积分等于该窗口的重量下降量。
L_force  爬得越陡越费油；判据成立处重量越大越费油。
L_phase  过渡点上同一状态经两侧相位子网络须给出一致的值，两台发动机分别比较。

三项物理惩罚只在训练分区上计算，推理阶段不做任何改正。
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from af_data import AIRCRAFT_FEATURE_NAMES, FEATURE_INDEX, aircraft_feature_matrix

# 标准大气与重力常数，只用于力判据
GRAVITY = 9.80665
GAS_CONSTANT_AIR = 287.05287
SEA_LEVEL_PRESSURE = 101325.0
SEA_LEVEL_TEMPERATURE = 288.15
ISA_LAPSE_RATE = 0.0065

# Oswald 效率因子。OpenAP 逐机型标定值为 A320 0.799、A320neo 0.807、A321 0.785。
DEFAULT_OSWALD_EFFICIENCY = 0.80


class LawBasedLoss(nn.Module):

    def __init__(self, x_scale: torch.Tensor,
                 w_mass: float = 0.10, w_force: float = 0.10, w_phase: float = 0.02,
                 mass_huber_delta: float = 0.10,
                 oswald_efficiency: float = DEFAULT_OSWALD_EFFICIENCY,
                 max_lift_coefficient: float = 2.5,
                 force_sign: float = 1.0,
                 aircraft_feature_table: Optional[torch.Tensor] = None):
        """aircraft_feature_table 不给时用四机型表；迁移实验会传入含目标机型的扩展表。"""
        super().__init__()
        if oswald_efficiency <= 0:
            raise ValueError('Oswald 效率必须为正')
        if max_lift_coefficient <= 0:
            raise ValueError('升力系数上限必须为正')
        if force_sign not in (1.0, -1.0):
            raise ValueError('force_sign 只能取 +1 或 -1')
        if min(w_mass, w_force, w_phase) < 0:
            raise ValueError('物理项权重不得为负')

        self.w_mass = float(w_mass)
        self.w_force = float(w_force)
        self.w_phase = float(w_phase)
        self.mass_huber_delta = float(mass_huber_delta)
        self.oswald_efficiency = float(oswald_efficiency)
        self.max_lift_coefficient = float(max_lift_coefficient)
        # +1 要求油耗随航迹角与重量上升，即物理给出的方向；
        # -1 反过来要求下降，用作对照：同样的惩罚项、同样的作用范围，只有方向是错的。
        self.force_sign = float(force_sign)

        table = (aircraft_feature_matrix() if aircraft_feature_table is None
                 else aircraft_feature_table.detach().clone().float())
        wing_area = table[:, AIRCRAFT_FEATURE_NAMES.index('wing_area')]
        wing_span = table[:, AIRCRAFT_FEATURE_NAMES.index('wing_span')]
        aspect_ratio = wing_span.square() / wing_area
        self.register_buffer('wing_area', wing_area)
        self.register_buffer('k_induced',
                             1.0 / (math.pi * aspect_ratio * self.oswald_efficiency))

        scale = x_scale.detach().clone().float()
        self.register_buffer('sd_fpa', scale[FEATURE_INDEX['FPA']].clamp_min(1e-8))
        self.register_buffer('sd_gwc', scale[FEATURE_INDEX['GWC']].clamp_min(1e-8))

    # ── 力判据 ──────────────────────────────────────────────────

    def gates(self, features: torch.Tensor, aircraft_ids: torch.Tensor
              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回两个条件。

        第一个是升力公式是否适用（`C_L < 2.5`）。整条受力推导都建立在升力公式上，
        公式不适用的记录两个方向都不施加惩罚。
        第二个是重量方向是否成立（`tanγ > −2kC_L`），只约束重量方向。
        """
        altitude = features[:, FEATURE_INDEX['ALT_QNH']]
        fpa = features[:, FEATURE_INDEX['FPA']]
        tas = features[:, FEATURE_INDEX['TAS']]
        temperature = features[:, FEATURE_INDEX['SAT']].clamp_min(150.0)
        mass = features[:, FEATURE_INDEX['GWC']]
        ids = aircraft_ids.long().view(-1)

        base = torch.clamp(1.0 - ISA_LAPSE_RATE * altitude / SEA_LEVEL_TEMPERATURE, min=0.05)
        pressure = SEA_LEVEL_PRESSURE * base.pow(GRAVITY / (GAS_CONSTANT_AIR * ISA_LAPSE_RATE))
        density = pressure / (GAS_CONSTANT_AIR * temperature)
        dynamic_pressure = 0.5 * density * tas.square()
        lift_coefficient = (mass * GRAVITY * torch.cos(fpa)
                            / dynamic_pressure.clamp_min(1.0) / self.wing_area[ids])

        lift_equation_applies = lift_coefficient < self.max_lift_coefficient
        weight_direction_holds = (
            torch.sin(fpa)
            + 2.0 * self.k_induced[ids] * lift_coefficient * torch.cos(fpa)) > 0
        return lift_equation_applies, weight_direction_holds

    def weight_gate(self, features: torch.Tensor, aircraft_ids: torch.Tensor) -> torch.Tensor:
        """重量方向实际生效的记录：升力公式适用，且判据成立。"""
        applies, holds = self.gates(features, aircraft_ids)
        return applies & holds

    def force_loss(self, model: nn.Module, features: torch.Tensor,
                   phases: torch.Tensor, aircraft_ids: torch.Tensor
                   ) -> Dict[str, torch.Tensor]:
        x = features.detach().clone().requires_grad_(True)
        return self.force_terms(model(x, phases, aircraft_ids), x, aircraft_ids)

    def force_terms(self, predictions: torch.Tensor, x: torch.Tensor,
                    aircraft_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """由已有的一次前向取两个方向导数，x 必须是带梯度的输入。"""
        total = predictions.sum(dim=1)
        grad, = torch.autograd.grad(total.sum(), x, create_graph=True)

        fpa = x[:, FEATURE_INDEX['FPA']]
        tas = x[:, FEATURE_INDEX['TAS']]
        # IVV 与 FPA 在几何上锁定，须取沿几何一致方向的方向导数
        d_gamma = (grad[:, FEATURE_INDEX['FPA']]
                   + tas * torch.cos(fpa) * grad[:, FEATURE_INDEX['IVV']])
        d_mass = grad[:, FEATURE_INDEX['GWC']]

        reference = total.detach().clamp_min(1e-6)
        rel_gamma = d_gamma * self.sd_fpa / reference
        rel_mass = d_mass * self.sd_gwc / reference

        applies, holds = self.gates(x.detach(), aircraft_ids)
        weight_gate = applies & holds
        # 升力公式不适用的记录，两个方向都不惩罚
        sign = self.force_sign
        loss_gamma = (F.relu(-sign * rel_gamma[applies]).mean() if bool(applies.any())
                      else rel_gamma.sum() * 0.0)
        loss_weight = (F.relu(-sign * rel_mass[weight_gate]).mean()
                       if bool(weight_gate.any()) else rel_mass.sum() * 0.0)
        return {'force': loss_gamma + loss_weight,
                'force_gamma': loss_gamma.detach(),
                'force_weight': loss_weight.detach(),
                'force_gate_fraction': weight_gate.float().mean().detach()}

    # ── 质量守恒 ────────────────────────────────────────────────

    def mass_loss(self, model: nn.Module, features: torch.Tensor, phases: torch.Tensor,
                  aircraft_ids: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        n, k, d = features.shape
        predicted = model(features.reshape(n * k, d), phases.reshape(-1),
                          aircraft_ids.reshape(-1)).sum(dim=1).reshape(n, k)
        dt = times[:, 1:] - times[:, :-1]
        burn = (0.5 * (predicted[:, 1:] + predicted[:, :-1]) * dt).sum(dim=1)
        observed = (features[:, 0, FEATURE_INDEX['GWC']]
                    - features[:, -1, FEATURE_INDEX['GWC']])
        residual = (burn - observed) / observed.detach().clamp_min(1.0)
        return F.huber_loss(residual, torch.zeros_like(residual),
                            delta=self.mass_huber_delta, reduction='mean')

    # ── 相位连续 ────────────────────────────────────────────────

    @staticmethod
    def phase_loss(model: nn.Module, features: torch.Tensor, aircraft_ids: torch.Tensor,
                   phase_left: torch.Tensor, phase_right: torch.Tensor) -> torch.Tensor:
        left = model(features, phase_left, aircraft_ids)
        right = model(features, phase_right, aircraft_ids)
        # 两台发动机分别比较后再平均，先求和会让一台升一台降互相抵消
        return ((left - right).abs() / left.detach().clamp_min(1e-6)).mean()

    # ── 合成 ────────────────────────────────────────────────────

    def forward(self, model: nn.Module, features: torch.Tensor, targets: torch.Tensor,
                phases: torch.Tensor, aircraft_ids: torch.Tensor,
                mass_batch: Optional[Tuple[torch.Tensor, ...]] = None,
                transition_batch: Optional[Tuple[torch.Tensor, ...]] = None
                ) -> Dict[str, torch.Tensor]:
        if self.w_force > 0:
            # 同一次前向同时供数据项与力方向导数
            x = features.detach().clone().requires_grad_(True)
            predictions = model(x, phases, aircraft_ids)
            force = self.force_terms(predictions, x, aircraft_ids)
        else:
            predictions = model(features, phases, aircraft_ids)
            force = None

        data_loss = F.mse_loss(predictions, targets)
        zero = data_loss.new_zeros(())
        result: Dict[str, torch.Tensor] = {'data': data_loss}
        total = data_loss

        if force is not None:
            result.update(force)
            total = total + self.w_force * force['force']
        else:
            result.update({'force': zero, 'force_gamma': zero,
                           'force_weight': zero, 'force_gate_fraction': zero})

        if self.w_mass > 0 and mass_batch is not None:
            result['mass'] = self.mass_loss(model, *mass_batch)
            total = total + self.w_mass * result['mass']
        else:
            result['mass'] = zero

        if self.w_phase > 0 and transition_batch is not None:
            result['phase'] = self.phase_loss(model, *transition_batch)
            total = total + self.w_phase * result['phase']
        else:
            result['phase'] = zero

        result['total'] = total
        return result

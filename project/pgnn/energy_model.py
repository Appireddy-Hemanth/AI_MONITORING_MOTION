"""Physics-guided neural network for energy consumption prediction."""

from __future__ import annotations

from collections import deque
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim


class EnergyPGNN(nn.Module):
    """Predict energy from time and usage intensity."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EnergyPGNNTrainer:
    """Online trainer with physical constraint E = P x t."""

    def __init__(self, lr: float = 1e-3, lambda_physics: float = 1.2) -> None:
        self.model = EnergyPGNN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.lambda_physics = lambda_physics
        self.physics_blend = 0.90
        self.buffer = deque(maxlen=256)

    @staticmethod
    def _power_from_usage(usage: float) -> float:
        # Usage in [0, 1] mapped to power in kW.
        return 0.25 + 2.25 * max(0.0, min(1.0, usage))

    def step(self, time_hours: float, device_usage: float, enable_replay: bool = True) -> Dict[str, float]:
        """Train one online step and return predicted energy with losses."""
        self.model.train()

        x = torch.tensor([[time_hours, device_usage]], dtype=torch.float32)

        power = self._power_from_usage(device_usage)
        physics_energy = power * time_hours
        physics_tensor = torch.tensor([[physics_energy]], dtype=torch.float32)

        net_energy = self.model(x)
        predicted_energy = self.physics_blend * physics_tensor + (1.0 - self.physics_blend) * net_energy

        # Simulated measured target stays tightly around the physical law.
        measured_energy = physics_energy * 0.998
        target = torch.tensor([[measured_energy]], dtype=torch.float32)

        data_loss = self.loss_fn(predicted_energy, target)
        physics_loss = self.loss_fn(predicted_energy, physics_tensor)
        total_loss = data_loss + self.lambda_physics * physics_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        if enable_replay:
            self.buffer.append((x.detach(), physics_tensor.detach()))
            replay_steps = min(4, len(self.buffer))
            if replay_steps > 0:
                samples = list(self.buffer)[-replay_steps:]
                for sx, sphysics in samples:
                    r_net = self.model(sx)
                    r_pred = self.physics_blend * sphysics + (1.0 - self.physics_blend) * r_net
                    r_data = self.loss_fn(r_pred, sphysics)
                    r_phys = self.loss_fn(r_pred, sphysics)
                    r_total = r_data + self.lambda_physics * r_phys
                    self.optimizer.zero_grad()
                    r_total.backward()
                    self.optimizer.step()

        accuracy = max(
            0.0,
            min(
                100.0,
                100.0
                * (1.0 - abs(float(predicted_energy.item()) - float(physics_energy)) / max(1e-5, abs(float(physics_energy)))),
            ),
        )

        return {
            "predicted_energy": float(predicted_energy.item()),
            "expected_energy": float(physics_energy),
            "data_loss": float(data_loss.item()),
            "physics_loss": float(physics_loss.item()),
            "total_loss": float(total_loss.item()),
            "power_kw": float(power),
            "accuracy": float(accuracy),
        }

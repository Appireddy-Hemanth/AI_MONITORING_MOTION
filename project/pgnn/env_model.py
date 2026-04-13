"""Physics-guided neural network for environmental state prediction."""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


class EnvironmentPGNN(nn.Module):
    """Predicts normalized environmental values from sensor inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EnvironmentPGNNTrainer:
    """Online trainer with physics loss penalizing sudden prediction jumps."""

    def __init__(self, lr: float = 8e-4, lambda_physics: float = 0.5) -> None:
        self.model = EnvironmentPGNN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.lambda_physics = lambda_physics
        self.prev_prediction: Optional[torch.Tensor] = None
        self.prev_sensor: Optional[torch.Tensor] = None
        self.buffer: Deque[Tuple[torch.Tensor, torch.Tensor]] = deque(maxlen=256)
        self.physics_blend = 0.82

    @staticmethod
    def _adaptive_blend(base_blend: float, x: torch.Tensor, prior: torch.Tensor) -> float:
        """Reduce smoothing when sensor values shift abruptly, to avoid laggy predictions."""
        delta = float(torch.mean(torch.abs(x - prior)).item())
        adjusted = base_blend - 0.35 * delta
        return max(0.68, min(base_blend, adjusted))

    @staticmethod
    def _normalize(values: Dict[str, float]) -> torch.Tensor:
        # Approximate scaling ranges for stable training.
        t = values["temperature"] / 60.0
        h = values["humidity"] / 100.0
        g = values["gas"] / 1000.0
        return torch.tensor([[t, h, g]], dtype=torch.float32)

    @staticmethod
    def _denormalize(x: torch.Tensor) -> Dict[str, float]:
        return {
            "temperature": float(x[0, 0].item() * 60.0),
            "humidity": float(x[0, 1].item() * 100.0),
            "gas": float(x[0, 2].item() * 1000.0),
        }

    @staticmethod
    def _vector_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
        # 100*(1-MAPE) style bounded accuracy for readable live quality tracking.
        eps = 1e-5
        rel_err = torch.abs(pred - target) / torch.clamp(torch.abs(target), min=eps)
        score = 100.0 * (1.0 - float(torch.mean(rel_err).item()))
        return max(0.0, min(100.0, score))

    def step(self, sensor_values: Dict[str, float], enable_replay: bool = True) -> Dict[str, float]:
        """Train one online step and return predictions with loss components."""
        self.model.train()

        x = self._normalize(sensor_values)
        prior = x if self.prev_sensor is None else self.prev_sensor
        net_pred = self.model(x)
        blend = self._adaptive_blend(self.physics_blend, x, prior)
        pred = blend * prior + (1.0 - blend) * net_pred

        data_loss = self.loss_fn(pred, x)
        smooth_loss = self.loss_fn(pred, prior)

        if self.prev_prediction is None:
            jump_loss = torch.tensor(0.0, dtype=torch.float32)
        else:
            # Enforces temporal smoothness: consecutive predictions should not jump abruptly.
            jump_loss = torch.mean((pred - self.prev_prediction) ** 2)

        physics_loss = 0.6 * smooth_loss + 0.4 * jump_loss
        total_loss = data_loss + self.lambda_physics * physics_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Replay a few past points each step for quick convergence and reduced drift.
        if enable_replay:
            self.buffer.append((x.detach(), prior.detach()))
            replay_steps = min(4, len(self.buffer))
            if replay_steps > 0:
                samples = list(self.buffer)[-replay_steps:]
                for sx, sprior in samples:
                    r_net = self.model(sx)
                    r_blend = self._adaptive_blend(self.physics_blend, sx, sprior)
                    r_pred = r_blend * sprior + (1.0 - r_blend) * r_net
                    r_data = self.loss_fn(r_pred, sx)
                    r_phys = self.loss_fn(r_pred, sprior)
                    r_total = r_data + self.lambda_physics * r_phys
                    self.optimizer.zero_grad()
                    r_total.backward()
                    self.optimizer.step()

        self.prev_prediction = pred.detach()
        self.prev_sensor = x.detach()

        accuracy = self._vector_accuracy(pred.detach(), x)

        denorm = self._denormalize(pred.detach())
        denorm.update(
            {
                "data_loss": float(data_loss.item()),
                "physics_loss": float(physics_loss.item()),
                "total_loss": float(total_loss.item()),
                "accuracy": float(accuracy),
            }
        )
        return denorm

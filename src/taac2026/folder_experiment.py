from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import DataConfig, ModelConfig, TrainConfig


@dataclass(slots=True)
class FolderExperiment:
    name: str
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    build_data_pipeline: Callable[..., Any]
    build_model_component: Callable[..., Any]
    build_loss_stack: Callable[..., Any]
    build_optimizer_component: Callable[..., Any]
    switches: dict[str, bool] = field(default_factory=dict)

    def clone(self) -> "FolderExperiment":
        return FolderExperiment(
            name=self.name,
            data=deepcopy(self.data),
            model=deepcopy(self.model),
            train=deepcopy(self.train),
            build_data_pipeline=self.build_data_pipeline,
            build_model_component=self.build_model_component,
            build_loss_stack=self.build_loss_stack,
            build_optimizer_component=self.build_optimizer_component,
            switches=deepcopy(self.switches),
        )
from __future__ import annotations

import importlib

import pytest
import torch

from taac2026.experiment_loader import load_experiment_package
from tests.training_stack_support import TestWorkspace, prepare_experiment


@pytest.mark.parametrize(
    "module_path",
    [
        "config.gen.ctr_baseline",
        "config.gen.deepcontextnet",
        "config.gen.interformer",
        "config.gen.onetrans",
        "config.gen.hyformer",
        "config.gen.unirec",
        "config.gen.uniscaleformer",
        "config.gen.oo",
    ],
)
def test_experiment_package_builds_and_runs_forward(module_path: str, test_workspace: TestWorkspace) -> None:
    experiment = importlib.import_module(module_path).EXPERIMENT
    experiment = prepare_experiment(experiment, test_workspace)

    train_loader, _, data_stats = experiment.build_data_pipeline(
        experiment.data,
        experiment.model,
        experiment.train,
    )
    batch = next(iter(train_loader))
    model = experiment.build_model_component(experiment.data, experiment.model, data_stats.dense_dim)
    logits = model(batch)

    assert logits.shape == batch.labels.shape
    assert torch.isfinite(logits).all().item()


@pytest.mark.parametrize(
    "experiment_path",
    [
        "config/gen/ctr_baseline",
        "config/gen/deepcontextnet",
        "config/gen/unirec",
        "config/gen/uniscaleformer",
    ],
)
def test_experiment_package_directory_path_loads_namespace_relative_imports(experiment_path: str) -> None:
    experiment = load_experiment_package(experiment_path)

    assert experiment.name
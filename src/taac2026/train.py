from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from .experiment_loader import load_experiment_package
from .folder_experiment import FolderExperiment
from .metrics import compute_classification_metrics, percentile, safe_mean
from .utils import ensure_dir, write_json


def render_training_curves_plot(
    output_path: Path,
    train_losses: list[float],
    val_losses: list[float],
    val_aucs: list[float],
    best_epoch: int,
) -> None:
    if not train_losses:
        return

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to render training curves; run uv sync --locked") from exc

    epochs = list(range(1, len(train_losses) + 1))
    figure, (loss_axis, auc_axis) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    loss_axis.plot(epochs, train_losses, marker="o", linewidth=2.0, label="train_loss")
    loss_axis.plot(epochs, val_losses, marker="s", linewidth=2.0, label="val_loss")
    loss_axis.set_ylabel("Loss")
    loss_axis.grid(True, alpha=0.3)
    loss_axis.legend()

    auc_axis.plot(epochs, val_aucs, marker="o", linewidth=2.0, color="#2ca02c", label="val_auc")
    if 0 < best_epoch <= len(val_aucs):
        best_auc = val_aucs[best_epoch - 1]
        auc_axis.axvline(best_epoch, color="#7f7f7f", linestyle="--", linewidth=1.5, label=f"best_epoch={best_epoch}")
        auc_axis.scatter([best_epoch], [best_auc], color="#d62728", s=60, zorder=3, label=f"best_auc={best_auc:.4f}")
    auc_axis.set_xlabel("Epoch")
    auc_axis.set_ylabel("AUC")
    auc_axis.grid(True, alpha=0.3)
    auc_axis.legend()

    figure.suptitle("Training Curves")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def write_training_curve_artifacts(
    output_dir: Path,
    train_losses: list[float],
    val_losses: list[float],
    val_aucs: list[float],
    best_epoch: int,
) -> None:
    curves = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_auc": val_aucs,
        "best_epoch": best_epoch,
    }
    write_json(output_dir / "training_curves.json", curves)
    render_training_curves_plot(
        output_path=output_dir / "training_curves.png",
        train_losses=train_losses,
        val_losses=val_losses,
        val_aucs=val_aucs,
        best_epoch=best_epoch,
    )


def select_device(device_name: str | None = None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def collect_loader_outputs(model, loader, device, loss_fn=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    logits_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    group_list: list[np.ndarray] = []
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            if loss_fn is not None:
                losses.append(float(loss_fn(logits, batch.labels).detach().cpu().item()))
            logits_list.append(logits.detach().cpu().numpy())
            labels_list.append(batch.labels.detach().cpu().numpy())
            group_list.append(batch.user_indices.detach().cpu().numpy())
    if not logits_list:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty, empty, 0.0
    return (
        np.concatenate(logits_list, axis=0),
        np.concatenate(labels_list, axis=0),
        np.concatenate(group_list, axis=0),
        safe_mean(losses),
    )


def measure_latency(model, loader, device, warmup_steps: int, measure_steps: int) -> dict[str, float]:
    durations: list[float] = []
    model.eval()
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        if step < warmup_steps:
            with torch.no_grad():
                _ = model(batch)
            continue
        if measure_steps > 0 and len(durations) >= measure_steps:
            break
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        durations.append((elapsed * 1000.0) / max(batch.batch_size, 1))
    return {
        "mean_latency_ms_per_sample": safe_mean(durations),
        "p95_latency_ms_per_sample": percentile(durations, 95.0),
    }


def profiler_activities_for(device: torch.device) -> list[ProfilerActivity]:
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    return activities


def loader_num_batches(loader) -> int:
    try:
        return int(len(loader))
    except TypeError:
        return 0


def loader_num_samples(loader, max_batches: int | None = None) -> int:
    sample_count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        sample_count += int(batch.batch_size)
    return sample_count


def count_latency_probe_batches(total_batches: int, warmup_steps: int, measure_steps: int) -> int:
    if total_batches <= 0:
        return 0
    warmup_batches = min(total_batches, max(warmup_steps, 0))
    remaining_batches = max(total_batches - warmup_batches, 0)
    measured_batches = remaining_batches if measure_steps <= 0 else min(remaining_batches, measure_steps)
    return warmup_batches + measured_batches


def collect_model_profile(model, loader, device) -> dict[str, float | int | str]:
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())

    profile_batch = next(iter(loader), None)
    if profile_batch is None:
        return {
            "profile_scope": "single_eval_forward",
            "profile_batch_size": 0,
            "total_parameters": int(total_parameters),
            "trainable_parameters": int(trainable_parameters),
            "parameter_size_mb": parameter_bytes / (1024.0 * 1024.0),
            "flops_per_batch": 0.0,
            "tflops_per_batch": 0.0,
            "flops_per_sample": 0.0,
        }

    profile_batch = profile_batch.to(device)
    batch_size = max(profile_batch.batch_size, 1)
    was_training = model.training
    model.eval()

    with profile(activities=profiler_activities_for(device), with_flops=True, record_shapes=False, acc_events=True) as profiler:
        with torch.no_grad():
            _ = model(profile_batch)

    if was_training:
        model.train()

    total_flops = float(profiler.key_averages().total_average().flops or 0.0)
    return {
        "profile_scope": "single_eval_forward",
        "profile_batch_size": int(profile_batch.batch_size),
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "parameter_size_mb": parameter_bytes / (1024.0 * 1024.0),
        "flops_per_batch": total_flops,
        "tflops_per_batch": total_flops / 1.0e12,
        "flops_per_sample": total_flops / float(batch_size),
    }


def collect_compute_profile(
    experiment: FolderExperiment,
    model,
    loss_fn,
    train_loader,
    val_loader,
    data_stats,
    device,
    model_profile: dict[str, float | int | str],
) -> dict[str, float | int | str]:
    train_batches_per_epoch = loader_num_batches(train_loader)
    val_batches_per_epoch = loader_num_batches(val_loader)
    latency_probe_batches = count_latency_probe_batches(
        total_batches=val_batches_per_epoch,
        warmup_steps=experiment.train.latency_warmup_steps,
        measure_steps=experiment.train.latency_measure_steps,
    )
    latency_probe_samples = loader_num_samples(val_loader, max_batches=latency_probe_batches)

    train_profile_batch = next(iter(train_loader), None)
    if train_profile_batch is None:
        train_step_flops = 0.0
        train_profile_batch_size = 0
        train_step_flops_per_sample = 0.0
    else:
        train_profile_batch = train_profile_batch.to(device)
        train_profile_batch_size = int(train_profile_batch.batch_size)
        profile_model = experiment.build_model_component(experiment.data, experiment.model, data_stats.dense_dim)
        profile_model = profile_model.to(device)
        profile_optimizer = experiment.build_optimizer_component(profile_model, experiment.train)
        profile_model.train()

        with profile(
            activities=profiler_activities_for(device),
            with_flops=True,
            record_shapes=False,
            acc_events=True,
        ) as profiler:
            profile_optimizer.zero_grad(set_to_none=True)
            logits = profile_model(train_profile_batch)
            loss = loss_fn(logits, train_profile_batch.labels)
            loss.backward()
            if experiment.train.grad_clip_norm and experiment.train.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(profile_model.parameters(), experiment.train.grad_clip_norm)
            profile_optimizer.step()

        train_step_flops = float(profiler.key_averages().total_average().flops or 0.0)
        train_step_flops_per_sample = train_step_flops / float(max(train_profile_batch_size, 1))

        del profile_optimizer
        del profile_model
        del train_profile_batch
        if device.type == "cuda":
            torch.cuda.empty_cache()

    eval_flops_per_sample = float(model_profile.get("flops_per_sample", 0.0))
    train_samples_per_epoch = int(data_stats.train_size)
    val_samples_per_epoch = int(data_stats.val_size)

    estimated_train_flops_total = train_step_flops_per_sample * float(train_samples_per_epoch) * float(experiment.train.epochs)
    estimated_eval_flops_total = eval_flops_per_sample * float(val_samples_per_epoch) * float(experiment.train.epochs)
    estimated_latency_probe_flops_total = eval_flops_per_sample * float(latency_probe_samples)
    estimated_end_to_end_flops_total = (
        estimated_train_flops_total
        + estimated_eval_flops_total
        + estimated_latency_probe_flops_total
    )

    return {
        "estimation_method": "profiled_single_step_scaled_by_observed_sample_counts",
        "epochs": int(experiment.train.epochs),
        "train_batches_per_epoch": train_batches_per_epoch,
        "val_batches_per_epoch": val_batches_per_epoch,
        "train_samples_per_epoch": train_samples_per_epoch,
        "val_samples_per_epoch": val_samples_per_epoch,
        "latency_probe_batches": latency_probe_batches,
        "latency_probe_samples": latency_probe_samples,
        "train_profile_scope": "single_train_step_forward_backward_optimizer",
        "train_profile_batch_size": train_profile_batch_size,
        "train_step_flops": train_step_flops,
        "train_step_tflops": train_step_flops / 1.0e12,
        "train_step_flops_per_sample": train_step_flops_per_sample,
        "estimated_train_flops_total": estimated_train_flops_total,
        "estimated_train_tflops_total": estimated_train_flops_total / 1.0e12,
        "estimated_eval_flops_total": estimated_eval_flops_total,
        "estimated_eval_tflops_total": estimated_eval_flops_total / 1.0e12,
        "estimated_latency_probe_flops_total": estimated_latency_probe_flops_total,
        "estimated_latency_probe_tflops_total": estimated_latency_probe_flops_total / 1.0e12,
        "estimated_end_to_end_flops_total": estimated_end_to_end_flops_total,
        "estimated_end_to_end_tflops_total": estimated_end_to_end_flops_total / 1.0e12,
    }


def run_training(experiment: FolderExperiment) -> dict[str, Any]:
    output_dir = ensure_dir(experiment.train.output_dir)
    device = select_device(experiment.train.device)

    train_loader, val_loader, data_stats = experiment.build_data_pipeline(
        experiment.data,
        experiment.model,
        experiment.train,
    )
    model = experiment.build_model_component(experiment.data, experiment.model, data_stats.dense_dim)
    model = model.to(device)
    loss_fn, auxiliary_loss = experiment.build_loss_stack(
        experiment.data,
        experiment.model,
        experiment.train,
        data_stats,
        device,
    )
    optimizer = experiment.build_optimizer_component(model, experiment.train)
    model_profile = collect_model_profile(model, val_loader, device)
    compute_profile = collect_compute_profile(
        experiment=experiment,
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        data_stats=data_stats,
        device=device,
        model_profile=model_profile,
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_aucs: list[float] = []
    best_auc = float("-inf")
    best_epoch = 0
    best_metrics: dict[str, Any] = {}

    for epoch in range(1, experiment.train.epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = loss_fn(logits, batch.labels)
            if getattr(auxiliary_loss, "enabled", False) and getattr(auxiliary_loss, "requires_aux", False):
                raise RuntimeError("Auxiliary losses requiring extra tensors are not implemented")
            loss.backward()
            if experiment.train.grad_clip_norm and experiment.train.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), experiment.train.grad_clip_norm)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        train_loss = safe_mean(batch_losses)
        val_logits, val_labels, val_groups, val_loss = collect_loader_outputs(model, val_loader, device, loss_fn)
        val_metrics = compute_classification_metrics(val_labels, val_logits, val_groups)
        val_auc = float(val_metrics["auc"])

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)

        if val_auc >= best_auc:
            best_auc = val_auc
            best_epoch = epoch
            best_metrics = dict(val_metrics)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": val_metrics,
                },
                output_dir / "best.pt",
            )

        write_training_curve_artifacts(
            output_dir=output_dir,
            train_losses=train_losses,
            val_losses=val_losses,
            val_aucs=val_aucs,
            best_epoch=best_epoch,
        )

    latency = measure_latency(
        model,
        val_loader,
        device,
        warmup_steps=experiment.train.latency_warmup_steps,
        measure_steps=experiment.train.latency_measure_steps,
    )

    summary = {
        "model_name": experiment.model.name,
        "best_epoch": best_epoch,
        "best_val_auc": best_auc,
        "metrics": best_metrics,
        "model_profile": model_profile,
        "compute_profile": compute_profile,
        **latency,
    }

    write_json(output_dir / "summary.json", summary)
    return summary


def parse_train_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TAAC 2026 experiment")
    parser.add_argument("--experiment", required=True, help="Experiment package path or module path")
    parser.add_argument("--run-dir", help="Override output directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_train_args(argv)
    experiment = load_experiment_package(args.experiment)
    if args.run_dir:
        experiment.train.output_dir = args.run_dir
    summary = run_training(experiment)
    print(summary)
    return summary


if __name__ == "__main__":
    main()
import os
import pickle
import random
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.lines import Line2D
from omegaconf import OmegaConf
from scipy.io import loadmat
from sklearn.decomposition import PCA

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

import lfadsci.model
import lfadsci.run_utils
import lfadsci.shared_utils


def cut_go_epoch(neural_data: np.ndarray, go_epochs: np.ndarray) -> np.ndarray:
    trials = []
    for start_time, end_time in go_epochs:
        trials.append(neural_data[int(start_time):int(end_time), :])
    return np.asarray(trials)


def _resolve_channel_data(tx_data: np.ndarray, config) -> np.ndarray:
    channels_cfg = config.channels

    if isinstance(channels_cfg, str):
        if "channel_map" not in config or channels_cfg not in config.channel_map:
            raise ValueError(
                f"Unknown channels key '{channels_cfg}'. Expected one of: {list(config.get('channel_map', {}).keys())}"
            )
        channels_cfg = config.channel_map[channels_cfg]

    channels_np = np.asarray(channels_cfg, dtype=np.int32)

    if channels_np.ndim == 1 and channels_np.size == 2:
        start, stop = int(channels_np[0]), int(channels_np[1])
        return tx_data[:, start:stop]

    if channels_np.ndim == 2 and channels_np.shape[1] == 2:
        chunks = [tx_data[:, int(start):int(stop)] for start, stop in channels_np]
        return np.concatenate(chunks, axis=1)

    if channels_np.ndim == 1:
        return tx_data[:, channels_np.astype(np.int32)]

    raise ValueError(
        "Invalid channels format. Use a key in channel_map, [start, stop], "
        "[[start1, stop1], [start2, stop2]], or explicit indices [i1, i2, ...]."
    )


def build_t19_data(config):
    data_path = Path(config.dataset.data_dir) / config.dataset.data_file
    if not data_path.exists():
        raise FileNotFoundError(f"T19 data file not found: {data_path}")

    data_mat = loadmat(data_path)

    cues_trials = np.squeeze(data_mat[config.dataset.cue_key])
    cue_names = None
    if "cueList" in data_mat:
        raw_cue_list = np.squeeze(data_mat["cueList"])
        cue_names = []
        for item in raw_cue_list:
            if isinstance(item, np.ndarray):
                if item.size == 0:
                    cue_names.append("")
                    continue
                value = item.flat[0]
            else:
                value = item
            cue_names.append(str(value))

    num_trials = len(cues_trials)
    delays_trials = np.array([""] * num_trials)
    session_trials = np.ones((num_trials, 1), dtype=np.int32)

    go_epochs_raw = np.squeeze(data_mat[config.dataset.go_epoch_key][:, config.dataset.go_epoch_column])
    go_epochs_raw = go_epochs_raw.reshape((-1, 1))
    epoch_time_window = np.array(config.epoch_time_window)
    time_steps_before_go = int(round(epoch_time_window[0]))
    time_steps_after_go = int(round(epoch_time_window[1]))

    train_epochs = np.concatenate(
        [
            go_epochs_raw + time_steps_before_go,
            go_epochs_raw + time_steps_after_go,
        ],
        axis=1,
    )

    tx = _resolve_channel_data(data_mat[config.dataset.tx_key], config)

    neural_trials = cut_go_epoch(tx, train_epochs)

    neural_trials = neural_trials.astype(np.float32)

    data_list = [
        {
            "neural": neural_trials,
            "cues": cues_trials,
            "delays": delays_trials,
            "session_id": session_trials,
            "task": str(config.dataset.task_name),
        }
    ]

    datagenerator_combined, datasets_list = lfadsci.shared_utils.combine_datasets(
        data_list,
        batch_sz=int(config.batch_size),
        train_frac=float(config.train_frac),
        val_frac=float(config.val_frac),
        test_frac=float(config.test_frac),
        seed=int(config.data_seed),
        data_weight=[1.0],
    )

    data = {
        "datagenerator": datagenerator_combined,
        "datasets": datasets_list,
        "data_val_weight": [1.0],
        "cue_names": cue_names,
    }
    return data


def _sanitize_token(value) -> str:
    return str(value).replace("/", "-").replace(" ", "")


def _data_selection_tag(config) -> str:
    channels_cfg = config.channels
    if isinstance(channels_cfg, str):
        channels_token = channels_cfg
    else:
        channels_token = _sanitize_token(channels_cfg)

    epoch_window = np.asarray(config.epoch_time_window).astype(int).reshape(-1)
    if epoch_window.size >= 2:
        window_token = f"{epoch_window[0]}to{epoch_window[1]}"
    else:
        window_token = _sanitize_token(config.epoch_time_window)

    return "_".join(
        [
            f"channels{_sanitize_token(channels_token)}",
            f"window{_sanitize_token(window_token)}",
        ]
    )


def _partition_accuracy_text(partition_results: dict) -> str:
    metrics = partition_results.get("metrics", {})
    acc_items = sorted(
        [
            (k, v)
            for k, v in metrics.items()
            if k.startswith("cue_classification_acc_from_")
        ],
        key=lambda item: item[0],
    )
    if not acc_items:
        return "Accuracies: not available"

    formatted = [
        f"{key.replace('cue_classification_acc_from_', '')}: {float(value):.3f}"
        for key, value in acc_items
    ]
    return "Accuracies | " + " | ".join(formatted)


def _cue_label_from_trial_cue(cue_value, cue_names):
    if cue_names is not None:
        try:
            idx = int(cue_value) - 1
            if 0 <= idx < len(cue_names):
                return str(cue_names[idx])
        except Exception:
            pass
    return str(cue_value)


def _style_from_cue_label(label: str):
    label_up = label.upper()
    if "HAMMER" in label_up:
        color = "r"
    elif "KNIFE" in label_up:
        color = "g"
    elif "SCREWDRIVER" in label_up:
        color = "b"
    elif "SPOON" in label_up:
        color = "c"
    elif "TONGS" in label_up:
        color = "m"
    else:
        color = "0.4"

    if "PINCH" in label_up:
        linestyle = "--"
    else:
        linestyle = "-"

    return color, linestyle


def plot_state_pca_trajectories(results: dict, datasets: dict, config, cue_names=None):
    output_plot_dir = Path(config.outputPlotDir)
    output_plot_dir.mkdir(parents=True, exist_ok=True)
    data_tag = _data_selection_tag(config)

    for partition in ["train", "eval", "test"]:
        if partition not in results or "state" not in results[partition]:
            print(f"Skipping PCA plot for partition '{partition}' (state not available)")
            continue

        states = np.asarray(results[partition]["state"])
        if states.ndim != 3:
            print(f"Skipping PCA plot for partition '{partition}' (state shape {states.shape} unsupported)")
            continue

        cues_raw = np.asarray(datasets[partition]["cues"])

        n_trials, n_time_bins, n_factors = states.shape
        states_2d = states.reshape(-1, n_factors)
        pca = PCA(n_components=2)
        states_pca = pca.fit_transform(states_2d).reshape(n_trials, n_time_bins, 2)

        fig, ax = plt.subplots(figsize=(10, 8))

        for trial_idx in range(n_trials):
            cue_label = _cue_label_from_trial_cue(cues_raw[trial_idx], cue_names)
            color, linestyle = _style_from_cue_label(cue_label)
            ax.plot(
                states_pca[trial_idx, :, 0],
                states_pca[trial_idx, :, 1],
                color=color,
                linestyle=linestyle,
                alpha=0.35,
                linewidth=1.0,
            )
            ax.plot(
                states_pca[trial_idx, 0, 0],
                states_pca[trial_idx, 0, 1],
                marker="o",
                color=color,
                markersize=3,
            )

        object_handles = [
            Line2D([0], [0], color="r", lw=2, linestyle="-", label="HAMMER"),
            Line2D([0], [0], color="g", lw=2, linestyle="-", label="KNIFE"),
            Line2D([0], [0], color="b", lw=2, linestyle="-", label="SCREWDRIVER"),
            Line2D([0], [0], color="c", lw=2, linestyle="-", label="SPOON"),
            Line2D([0], [0], color="m", lw=2, linestyle="-", label="TONGS"),
        ]
        style_handles = [
            Line2D([0], [0], color="k", lw=2, linestyle="-", label="Grasp (solid)"),
            Line2D([0], [0], color="k", lw=2, linestyle="--", label="Pinch (dashed)"),
        ]
        legend_obj = ax.legend(
            handles=object_handles,
            title="Color = Object",
            loc="upper left",
            framealpha=1.0,
            facecolor="white",
        )
        ax.add_artist(legend_obj)
        ax.legend(
            handles=style_handles,
            title="Line Style = Grasp Type",
            loc="lower left",
            framealpha=1.0,
            facecolor="white",
        )

        title = (
            f"T19 state PCA trajectories ({partition}) | "
            f"n_steps={config.n_steps}, n_hidden_decode={config.model.n_hidden_decode}, "
            f"factors={config.model.factors}, ic_dim={config.model.ic_dim}, bias_dim={config.model.bias_dim}"
        )
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.3)

        acc_text = _partition_accuracy_text(results[partition])
        fig.text(0.5, 0.01, acc_text, ha="center", va="bottom", fontsize=9)
        fig.tight_layout(rect=[0.0, 0.05, 1.0, 1.0])

        filename = f"state_pca_trajectories_{partition}_{data_tag}.png"
        save_path = output_plot_dir / filename
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved PCA plot: {save_path}")


@hydra.main(config_path="configs", config_name="t19_partial", version_base=None)
def app(config):
    os.makedirs(config.outputDir, exist_ok=True)
    os.makedirs(config.outputPlotDir, exist_ok=True)

    data_tag = _data_selection_tag(config)
    log_path = Path(config.outputPlotDir) / f"{data_tag}.log"

    with open(log_path, "a", encoding="utf-8") as log_file:
        with redirect_stdout(log_file), redirect_stderr(log_file):
            print(f"Logging to: {log_path}")
            print(OmegaConf.to_yaml(config))

            if "gpuNumber" in config and config.gpuNumber is not None:
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpuNumber)

            np.random.seed(int(config.seed))
            tf.random.set_seed(int(config.seed))
            random.seed(int(config.seed))

            print("Preparing T19 data...")
            data = build_t19_data(config)
            n_channels = [dataset["train"]["neural"].shape[-1] for dataset in data["datasets"]]
            print(f"n_channels: {n_channels}")

            print("Building LFADS model...")
            model = lfadsci.model.load_model_from_config(
                config,
                n_channels,
                model_filename=f"{config.outputDir}/model",
            )

            if str(config.mode).lower() == "train":
                print("Starting training...")
                start_time = time.time()
                lfadsci.run_utils.train(
                    data["datagenerator"]["train"],
                    model,
                    data_test=[d["eval"] for d in data["datasets"]],
                    lr_init=float(config.lr_init),
                    lr_stop=float(config.lr_stop),
                    lams=None,
                    n_steps=int(config.n_steps),
                    to_plot=False,
                    kl_warmup_start=[int(config.model.kl_warmup_start)],
                    kl_warmup_end=[int(config.model.kl_warmup_end)],
                    decay_factor=float(config.decay_factor),
                    gradient_clipping_norm=0.1,
                    savefile=f"{config.outputDir}/model",
                    n_eval_samples=None,
                    patience_till_lr_decay=int(config.patience_till_lr_decay),
                    save_freq=int(config.save_freq),
                    data_val_weight=data["data_val_weight"],
                )
                elapsed_sec = time.time() - start_time
                print(f"Training completed in {elapsed_sec:.2f} seconds")
                print(f"Training summary | n_steps: {int(config.n_steps)} | elapsed_seconds: {elapsed_sec:.2f}")
                model.load(f"{config.outputDir}/model")
            else:
                print("Skipping training (mode != train), loading existing model checkpoint")
                model.load(f"{config.outputDir}/model")

            print("Compiling posterior-sampled results (partial)...")
            results_list = []
            for dataset in data["datasets"]:
                results = lfadsci.shared_utils.compile_results(
                    model,
                    dataset,
                    n_samples=int(config.results.n_samples),
                    compute_kinematic_r2=False,
                )
                results_list.append(results)

            partial_path = Path(config.outputDir) / f"results_partial_{data_tag}.pkl"
            with open(partial_path, "wb") as file_handle:
                pickle.dump({"results_list": results_list}, file_handle)

            print(f"Saved partial results to: {partial_path}")

            if len(results_list) > 0:
                print("Plotting state PCA trajectories for train/eval/test...")
                plot_state_pca_trajectories(results_list[0], data["datasets"][0], config, cue_names=data.get("cue_names"))


if __name__ == "__main__":
    app()
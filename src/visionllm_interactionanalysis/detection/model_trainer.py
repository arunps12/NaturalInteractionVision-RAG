"""Stage 4 — Faster R-CNN Model Trainer.

Trains a Faster R-CNN (ResNet-50 FPN) on cleaned COCO data with:
- Config-driven hyper-parameters (configs/model.yaml)
- Optional Optuna HPO
- MLflow tracking via DagsHub
- Best + last checkpoint saving
"""

from __future__ import annotations

import os
import time
from typing import Any

import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm.auto import tqdm

from visionllm_interactionanalysis.config.artifacts import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from visionllm_interactionanalysis.config.settings import ModelTrainerConfig
from visionllm_interactionanalysis.utils import (
    PipelineException,
    ensure_dir,
    get_logger,
    load_json,
    read_yaml,
    write_yaml,
)
from visionllm_interactionanalysis.utils.ml_utils import (
    collate_fn,
    get_device,
    save_checkpoint,
    set_seed,
)

logger = get_logger(__name__)


# ── Dataset ──────────────────────────────────────────────────────────
class COCODetectionDataset(Dataset):
    """Minimal COCO detection dataset from a cleaned manifest."""

    def __init__(self, images_dir: str, ann_file: str, max_images: int | None = None):
        self.images_dir = images_dir
        coco = load_json(ann_file)

        self.images = coco.get("images", [])
        self.annotations = coco.get("annotations", [])
        self.categories = coco.get("categories", [])

        if not self.images or not self.annotations:
            raise PipelineException(f"Empty COCO JSON: {ann_file}")

        cat_ids = (
            sorted({c["id"] for c in self.categories})
            if self.categories
            else sorted({a["category_id"] for a in self.annotations})
        )
        self.cat_map = {cid: i + 1 for i, cid in enumerate(cat_ids)}

        self.img_info = {im["id"]: im for im in self.images}
        self.ann_by_img: dict[int, list[dict]] = {}
        for a in self.annotations:
            self.ann_by_img.setdefault(a["image_id"], []).append(a)

        self.ids = [im["id"] for im in self.images if im["id"] in self.ann_by_img]
        if max_images is not None:
            self.ids = self.ids[:max_images]
        if not self.ids:
            raise PipelineException("No images with annotations found")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        iid = self.ids[idx]
        info = self.img_info[iid]
        path = os.path.join(self.images_dir, info["file_name"])

        bgr = cv2.imread(path)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        img = torch.from_numpy(rgb).permute(2, 0, 1)

        boxes, labels, area, iscrowd = [], [], [], []
        for a in self.ann_by_img.get(iid, []):
            x, y, w, h = a["bbox"]
            if w <= 1 or h <= 1:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_map[a["category_id"]])
            area.append(float(a.get("area", w * h)))
            iscrowd.append(int(a.get("iscrowd", 0)))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([iid], dtype=torch.int64),
            "area": torch.tensor(area, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
        }
        return img, target


# ── Model factory ────────────────────────────────────────────────────
def build_fasterrcnn(num_classes: int, weights: str = "DEFAULT") -> torch.nn.Module:
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    return model


# ── Training loops ───────────────────────────────────────────────────
def _train_epoch(model, loader, optimizer, device, log_every: int = 50) -> float:
    model.train()
    total = 0.0
    for step, (imgs, targets) in enumerate(tqdm(loader, desc="Train", leave=False), 1):
        imgs = [i.to(device) for i in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        losses = model(imgs, targets)
        loss = sum(losses.values())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu())
    return total / max(len(loader), 1)


@torch.no_grad()
def _val_epoch(model, loader, device) -> float:
    model.train()  # Faster R-CNN requires train mode to compute losses
    total = 0.0
    for imgs, targets in tqdm(loader, desc="Val", leave=False):
        imgs = [i.to(device) for i in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        losses = model(imgs, targets)
        total += float(sum(losses.values()).detach().cpu())
    return total / max(len(loader), 1)


# ── ModelTrainer ─────────────────────────────────────────────────────
class ModelTrainer:
    """Trains Faster R-CNN with optional Optuna HPO + MLflow logging."""

    def __init__(
        self,
        config: ModelTrainerConfig,
        transformation_artifact: DataTransformationArtifact,
    ) -> None:
        self.cfg = config
        self.transform = transformation_artifact

    def _setup_mlflow(self, model_cfg: dict):
        mlflow_cfg = model_cfg.get("mlflow", {}) or {}
        if not mlflow_cfg.get("enabled", False):
            logger.info("MLflow disabled")
            return None
        try:
            from dotenv import load_dotenv
            load_dotenv(override=False)
            import dagshub  # noqa: F811
            import mlflow  # noqa: F811
            dagshub.init(
                repo_owner=os.environ.get("DAGSHUB_REPO_OWNER", "arunps12"),
                repo_name=os.environ.get("DAGSHUB_REPO_NAME", "VisionLLM_InteractionAnalysis"),
                mlflow=True,
            )
            exp_name = mlflow_cfg.get("experiment_name") or model_cfg.get("experiment", {}).get("name", "default")
            mlflow.set_experiment(exp_name)
            logger.info("MLflow enabled via DagsHub — experiment=%s", exp_name)
            return mlflow
        except Exception as e:
            logger.warning("MLflow setup failed: %s — continuing without tracking", e)
            return None

    def run(self) -> ModelTrainerArtifact:
        try:
            logger.info("===== Stage 4: Model Trainer Started =====")
            ensure_dir(self.cfg.stage_dir)

            model_cfg = read_yaml(self.cfg.model_config_file)
            manifest = read_yaml(self.transform.training_manifest_file)

            # Parse YAML config
            exp_cfg = model_cfg.get("experiment", {}) or {}
            run_prefix = exp_cfg.get("run_name_prefix", "run")

            t_cfg = model_cfg.get("training", {}) or {}
            seed = int(t_cfg.get("seed", 42))
            set_seed(seed)
            device = get_device(t_cfg.get("device", "auto"))
            logger.info("Device: %s", device)

            epochs = int(t_cfg.get("epochs", 6))
            num_workers = int(t_cfg.get("num_workers", 4))
            pin_memory = bool(t_cfg.get("pin_memory", True)) and device.type == "cuda"

            opt_cfg = t_cfg.get("optimizer", {}) or {}
            base_lr = float(opt_cfg.get("lr", 1e-4))
            weight_decay = float(opt_cfg.get("weight_decay", 0.0))

            dl_cfg = t_cfg.get("dataloader", {}) or {}
            base_bs = int(dl_cfg.get("batch_size", 4))

            dbg = t_cfg.get("debug", {}) or {}
            debug = bool(dbg.get("enabled", False))
            max_train = dbg.get("max_train_images") if debug else None
            max_val = dbg.get("max_val_images") if debug else None

            log_cfg = model_cfg.get("logging", {}) or {}
            log_every = int(log_cfg.get("log_every_n_steps", 50))
            save_every = int(log_cfg.get("save_every_n_epochs", 1))

            mcfg = model_cfg.get("model", {}) or {}
            weights = mcfg.get("weights", "DEFAULT")
            num_classes = int(mcfg.get("num_classes", 81))

            out_cfg = model_cfg.get("outputs", {}) or {}
            ckpt_name = out_cfg.get("checkpoint_name", "fasterrcnn_best.pt")
            best_path = os.path.join(self.cfg.stage_dir, ckpt_name)
            last_path = self.cfg.last_model_file

            # Datasets
            train_ds = COCODetectionDataset(
                manifest["paths"]["train_images"],
                manifest["annotations"]["train"],
                max_images=max_train,
            )
            val_ds = COCODetectionDataset(
                manifest["paths"]["val_images"],
                manifest["annotations"]["val"],
                max_images=max_val,
            )

            mlflow = self._setup_mlflow(model_cfg)

            # ── Training function ────────────────────────────────────
            best_val = float("inf")
            best_params: dict[str, Any] = {}

            def _train_run(bs: int, lr: float, trial_num: int | None) -> float:
                nonlocal best_val, best_params

                train_loader = DataLoader(
                    train_ds, batch_size=bs, shuffle=True,
                    num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory,
                )
                val_loader = DataLoader(
                    val_ds, batch_size=bs, shuffle=False,
                    num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory,
                )

                model = build_fasterrcnn(num_classes, weights).to(device)
                optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

                run = None
                if mlflow:
                    name = f"{run_prefix}_bs{bs}_lr{lr:.2e}"
                    if trial_num is not None:
                        name += f"_t{trial_num}"
                    run = mlflow.start_run(run_name=name)
                    mlflow.log_params({
                        "batch_size": bs, "lr": lr, "epochs": epochs,
                        "num_classes": num_classes, "seed": seed,
                    })

                run_best = float("inf")
                for ep in range(1, epochs + 1):
                    tr = _train_epoch(model, train_loader, optim, device, log_every)
                    va = _val_epoch(model, val_loader, device)
                    logger.info("[trial=%s] epoch %d/%d  train=%.6f  val=%.6f", trial_num, ep, epochs, tr, va)
                    if mlflow:
                        mlflow.log_metrics({"train_loss": tr, "val_loss": va}, step=ep)
                    if va < run_best:
                        run_best = va
                    if save_every and ep % save_every == 0:
                        tag = f"trial_{trial_num}" if trial_num is not None else "single"
                        save_checkpoint(model, os.path.join(self.cfg.stage_dir, f"{tag}_ep{ep}.pt"),
                                        extra={"epoch": ep, "val_loss": va, "lr": lr, "bs": bs})

                # Global best
                if run_best < best_val:
                    best_val = run_best
                    best_params = {"batch_size": bs, "lr": lr}
                    save_checkpoint(model, best_path, extra={"best_val_loss": run_best, "lr": lr, "bs": bs})

                save_checkpoint(model, last_path, extra={"val_loss": run_best, "lr": lr, "bs": bs})

                if mlflow and run:
                    mlflow.log_metric("best_val_loss", run_best)
                    mlflow.end_run()
                return run_best

            # ── HPO with Optuna ──────────────────────────────────────
            hpo_cfg = model_cfg.get("hpo", {}) or {}
            hpo_enabled = bool(hpo_cfg.get("enabled", False))

            if hpo_enabled:
                import optuna
                n_trials = int(hpo_cfg.get("n_trials", 10))
                sampler_seed = int(hpo_cfg.get("sampler", {}).get("seed", seed))
                study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=sampler_seed))
                ss = hpo_cfg.get("search_space", {}) or {}

                def objective(trial: optuna.Trial) -> float:
                    bs = trial.suggest_categorical("batch_size", ss.get("batch_size", {}).get("values", [2, 4]))
                    lr_low = float(ss.get("lr", {}).get("low", 1e-5))
                    lr_high = float(ss.get("lr", {}).get("high", 1e-3))
                    lr = trial.suggest_float("lr", lr_low, lr_high, log=True)
                    return _train_run(int(bs), float(lr), trial.number)

                study.optimize(objective, n_trials=n_trials, timeout=hpo_cfg.get("timeout_seconds"))
                best_params = dict(study.best_params)
                n_trials_done = n_trials
            else:
                _train_run(base_bs, base_lr, None)
                n_trials_done = 0

            # ── Report ───────────────────────────────────────────────
            report = {
                "status": "SUCCESS",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "device": str(device),
                "model": {"type": mcfg.get("type", "fasterrcnn_resnet50_fpn"), "num_classes": num_classes},
                "hpo": {
                    "enabled": hpo_enabled, "n_trials": n_trials_done,
                    "best_params": best_params, "best_val_loss": best_val,
                },
            }
            write_yaml(self.cfg.report_file, report)
            logger.info("===== Stage 4: Model Trainer Complete =====")

            return ModelTrainerArtifact(
                stage_dir=self.cfg.stage_dir,
                best_model_path=best_path,
                last_model_path=last_path,
                training_report_path=self.cfg.report_file,
                best_metric_name="val_loss",
                best_metric_value=best_val,
                hpo_enabled=hpo_enabled,
                best_params=best_params,
                n_trials=n_trials_done,
            )
        except PipelineException:
            raise
        except Exception as e:
            raise PipelineException("Model training failed", e)

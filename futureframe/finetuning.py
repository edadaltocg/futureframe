import os
import time
from collections import defaultdict
from typing import Optional

import pandas as pd
import torch.utils.data
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from futureframe.data.features import get_num_classes, prepare_target_for_eval
from futureframe.data.tabular_datasets import SupervisedDatasetWithInputEncoder
from futureframe.encoding.base import BaseFeatureEncoder
from futureframe.logger import logger as log
from futureframe.optim import get_linear_warmup_cos_lr_scheduler
from futureframe.tasks import create_task
from futureframe.utils import get_num_parameters, seed_all, send_to_device_recursively

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def finetune(
    model: nn.Module,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    num_classes: Optional[int] = None,
    max_steps: int = 1000,
    max_val_steps: int = 100,
    checkpoints_dir: Optional[str] = None,
    input_encoder: Optional[BaseFeatureEncoder] = None,
    num_eval: int = 10,
    patience: Optional[int] = 3,
    lr: float = 1e-3,
    batch_size: int = 64,
    num_workers: int = 0,
    val_size: float = 0.05,
    seed: int = 42,
):
    """
    Fine-tunes a model on the given training data.

    Parameters:
        model (torch.nn.Module): The model to be fine-tuned.
        X_train (pd.DataFrame): The input training data.
        y_train (pd.Series): The target labels for the training data.
        num_classes (int): The number of output classes.
        max_steps (int): The maximum number of training steps.
        max_val_steps (int): The maximum number of validation steps.
        checkpoints_dir (str, optional): Directory to save the best model checkpoints.
        input_encoder (BaseFeatureEncoder, optional): The model input encoder.
        num_eval (int, optional): Number of evaluations during training.
        patience (int, optional): Number of evaluations to wait for improvement before early stopping.
        lr (float, optional): Learning rate for the optimizer.
        batch_size (int, optional): Batch size for data loading.
        num_workers (int, optional): Number of worker threads for data loading.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        tuple: The fine-tuned model and the training history.
    """
    seed_all(seed)
    device = next(model.parameters()).device
    log.info(f"Using device: {device}")

    if num_classes is None:
        num_classes = get_num_classes(y_train)

    task = create_task(num_classes)
    log.info(f"Task: {task}")

    if input_encoder is not None:
        log.info("Fitting input encoder...")
        input_encoder.prepare(X_train)
        log.info("Input encoder fitted")

    log.info("Preparing data...")
    y_train = prepare_target_for_eval(y_train, num_classes=num_classes)
    log.info("Target prepared")
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=seed)
    log.info("Data split")
    train_dataset = SupervisedDatasetWithInputEncoder(X_train, y_train, input_encoder)
    val_dataset = SupervisedDatasetWithInputEncoder(X_val, y_val, input_encoder)
    log.info("Datasets created")
    log.info("Preparing data loaders...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
    )
    train_terator = iter(train_dataloader)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        pin_memory=True,
    )
    log.info("Data loaders created")

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_warmup_cos_lr_scheduler(optimizer, max_steps, lr=lr)

    trainable, non_trainable = get_num_parameters(model)
    log.debug(f"Trainable parameters: {trainable}, Non-trainable parameters: {non_trainable}")

    history = defaultdict(list)
    pbar = tqdm(range(max_steps), desc="Fine-tuning")
    eval_freq = max_steps // num_eval
    best_eval_metric = 1e18 if task.less_is_better else -1e18
    if patience is None:
        patience = max_steps
    patience_steps = patience * eval_freq
    patience_counter = 0
    best_model_state = model.state_dict()

    for i in pbar:
        model.train()
        try:
            x, y = next(train_terator)
        except StopIteration:
            train_terator = iter(train_dataloader)
            x, y = next(train_terator)

        assert len(y) > 0, "y is empty."

        t0_global = time.perf_counter()
        x = send_to_device_recursively(x, device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        t0 = time.perf_counter()
        optimizer.zero_grad()
        logits = model(x)
        loss = task.compute_loss(y, logits).mean()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        t1 = time.perf_counter()
        t_train = t1 - t0

        history["t/loss"].append(loss.item())

        # validation step TODO: replace with predict function
        if i % eval_freq == 0:
            y_pred, y_true = [], []
            model.eval()
            t0 = time.perf_counter()
            y_pred, y_true = [], []
            for j, (x, y) in tqdm(
                enumerate(val_dataloader),
                total=min(max_val_steps, len(val_dataloader)),
                desc="Evaluating",
                position=1,
            ):
                assert len(y) > 0, "y is empty."
                x = send_to_device_recursively(x, device)
                y = y.to(device)
                t0 = time.perf_counter()
                with torch.no_grad():
                    logits = model(x)
                t1 = time.perf_counter()
                loss = task.compute_loss(y, logits).mean()
                it_t = t1 - t0
                # loss = criterion(logits.squeeze(), y.squeeze()).mean()

                history["v/loss"].append(loss.item())

                y_pred.append(logits)
                y_true.append(y)

                if j >= max_val_steps:
                    break

            forward_t = it_t * len(val_dataloader)
            t1 = time.perf_counter()

            t0 = time.perf_counter()
            y_true = torch.cat(y_true, dim=0).squeeze().cpu().numpy()
            y_pred = torch.cat(y_pred, dim=0).squeeze().cpu().numpy()
            t1 = time.perf_counter()

            t0 = time.perf_counter()
            metrics = task.evaluate(y_true, y_pred)
            t1 = time.perf_counter()
            t_eval = t1 - t0
            log.debug(f"{forward_t=}, {t_eval=}")
            # TODO: put it to the task class
            best_metric_value = metrics[task.best_metric]
            if task.less_is_better:
                is_best = best_metric_value < best_eval_metric
            else:
                is_best = best_metric_value > best_eval_metric
            if is_best:
                patience_counter = 0
                best_eval_metric = best_metric_value
                best_model_state = model.state_dict()
                if checkpoints_dir is not None:
                    path = os.path.join(checkpoints_dir, "best_model.pth")
                    torch.save(best_model_state, path)
                    log.info(f"Saved best model to {path}.")

            for k in metrics:
                history[f"v/{k}"].append(metrics[k])
            history[f"best/{task.best_metric}"].append(best_eval_metric)
            history["pc"].append(patience_counter)

        t1_global = time.perf_counter()
        t_global = t1_global - t0_global
        latest_history = {k: v[-1] for k, v in history.items()}
        # time_dict = {"t/global": t_global, "t/train": t_train}
        log_dict = {**latest_history}
        pbar.set_postfix(**log_dict)
        # log.info(log_dict)

        patience_counter += 1
        if patience_counter >= patience_steps:
            log.info(f"Early stopping at step {i}.")
            break

        if i >= max_steps:
            break

    model.load_state_dict(best_model_state)

    return model, history
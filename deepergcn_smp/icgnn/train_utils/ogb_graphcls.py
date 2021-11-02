import torch
import numpy as np
import time

from torch.utils.tensorboard import SummaryWriter


def train_eval_model(
    model,
    loaders,
    optim,
    evaluator,
    max_epochs,
    device,
    task_type,
    eval_metric,
    criterion,
    warmup=None,
    scheduler=None,
    min_lr=None,
    max_hours=None,
    logdir="logs/output",
):
    best_val = np.inf if task_type == "regression" else 0

    start_time = time.time()

    writer = SummaryWriter(logdir) if logdir else None

    for epoch in range(max_epochs):
        # check if we need to stop after N hours
        t = time.time()
        if max_hours and t > start_time + (max_hours * 60 * 60):
            print(f"Stopping after time: {t - start_time}")
            break

        if warmup is not None:
            warmup.step(epoch + 1)

        train_loss, train_result = step(
            True,
            model,
            device,
            loaders["train"],
            criterion,
            evaluator,
            optim,
        )
        val_loss, val_result = step(
            False, model, device, loaders["val"], criterion, evaluator
        )

        if scheduler:
            current_lr = optim.param_groups[0]["lr"]
            if writer:
                writer.add_scalar("lr", current_lr, epoch)

            # check if lr is below the limit
            if current_lr < min_lr:
                # finish training
                break

            scheduler.step(val_loss)

        train_metric, val_metric = train_result[eval_metric], val_result[eval_metric]

        if writer:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("metric/train", train_metric, epoch)
            writer.add_scalar("metric/val", val_metric, epoch)

        print(
            f"Epoch:{epoch} tloss: {train_loss:.4f} vloss: {val_loss:.4f} tmetric: {train_metric:.4f} vmetric: {val_metric:.4f}"
        )

        # check if the val metric improved
        # update the val and test metrics
        # use the single display value to compare validation metric
        # cant compare an array of metrics
        if (task_type != "regression" and val_metric > best_val) or (
            val_metric < best_val
        ):
            best_val = val_metric
            final_test = evaluate(
                model, device, loaders["test"], evaluator,
            )[eval_metric]

            print(f"Val metric improved, test metric now: {final_test:.4f}")
            if writer:
                writer.add_scalar("metric/test", final_test, epoch)

    return {"best_val": best_val, "final_test": final_test}


def step(
    is_train,
    model,
    device,
    loader,
    criterion,
    evaluator,
    optimizer=None,
):
    loss_list = []
    y_true = []
    y_pred = []

    if is_train:
        model.train()
    else:
        model.eval()

    for _, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            print("Skipping batch")
            continue

        if is_train:
            optimizer.zero_grad()
        # indices to consider in this batch
        # if there are multiple targets, make it a single list of indices
        is_labeled = (batch.y == batch.y).all(dim=1)

        # can have multiple tasks here
        y = batch.y[is_labeled].to(torch.float32).view(batch.y.shape)

        # select only these predictions
        pred = model(batch)
        loss = criterion(pred.to(torch.float32)[is_labeled], y)

        if is_train:
            loss.backward()
            optimizer.step()

        # target is same shape as pred
        y_eval = batch.y.view(pred.shape)
        # pred is unchanged
        pred_eval = pred

        y_true.append(y_eval.detach().cpu())
        y_pred.append(pred_eval.detach().cpu())

        loss_list.append(loss.item())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return np.mean(loss_list), evaluator.eval(input_dict)


@torch.no_grad()
def evaluate(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for _, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            print("Skipping batch")
            pass
        else:
            pred = model(batch)

            # target is same shape as pred
            y = batch.y.view(pred.shape)
            # pred is unchanged

            y_true.append(y.detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)

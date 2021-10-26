import torch
import numpy as np
from collections import defaultdict
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
    multi_class,
    warmup=None,
    scheduler=None,
    min_lr=None,
    max_hours=None,
    logdir="logs/output",
):
    losses, metrics = defaultdict(list), defaultdict(list)
    best_val = np.inf if task_type == "regression" else 0

    start_time = time.time()

    writer = SummaryWriter(logdir)

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
            multi_class,
            criterion,
            evaluator,
            optim,
        )
        val_loss, val_result = step(
            False, model, device, loaders["val"], multi_class, criterion, evaluator
        )

        if scheduler:
            current_lr = optim.param_groups[0]["lr"]
            writer.add_scalar("lr", current_lr, epoch)

            # check if lr is below the limit
            if current_lr < min_lr:
                # finish training
                break

            scheduler.step(val_loss)

        train_metric, val_metric = train_result[eval_metric], val_result[eval_metric]

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
                model, device, loaders["test"], evaluator, multi_class
            )[eval_metric]

            print(f"Val metric improved, test metric now: {final_test:.4f}")
            writer.add_scalar("metric/test", final_test, epoch)

    return {"best_val": best_val, "final_test": final_test}


def step(
    is_train,
    model,
    device,
    loader,
    multi_class,
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

        # multi class, single prediction per graph
        if multi_class:
            # 1-dim index
            is_labeled = is_labeled.view(-1)
            # 1-dim target
            y = batch.y[is_labeled].view(-1)
        else:
            # can have multiple tasks here
            # TODO: dataset-specific way of handling?
            y = batch.y[is_labeled].to(torch.float32).view(batch.y.shape)

        # select only these predictions
        pred = model(batch)
        loss = criterion(pred.to(torch.float32)[is_labeled], y)

        if is_train:
            loss.backward()
            optimizer.step()

        # update y lists for evaluation
        if multi_class:
            # target is (N, 1)
            y_eval = batch.y.view(-1, 1)
            # take argmax over classes to get the prediction
            pred_eval = torch.argmax(pred, dim=1).view(-1, 1)
        else:
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


def train(model, device, loader, optimizer, criterion, multi_class, pred_list):
    loss_list = []
    model.train()

    for _, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            print("Skipping batch")
            pass
        else:
            optimizer.zero_grad()
            # indices to consider in this batch
            is_labeled = batch.y == batch.y

            if pred_list:
                pass
            # multi class, single prediction per graph
            elif multi_class:
                # 1-dim index
                is_labeled = is_labeled.view(-1)
                # 1-dim target
                y = batch.y[is_labeled].view(-1)
            else:
                # can have multiple tasks here
                y = batch.y[is_labeled].to(torch.float32)

            # select only these predictions
            pred = model(batch)

            # prediction is multi class, multi task?
            if pred_list:
                loss = 0
                for i in range(len(pred)):
                    loss += criterion(pred[i].to(torch.float32), batch.y_arr[:, i])
                loss = loss / len(pred)
            else:
                loss = criterion(pred.to(torch.float32)[is_labeled], y)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
    return np.mean(loss_list)


@torch.no_grad()
def evaluate(model, device, loader, evaluator, multi_class):
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

            if multi_class:
                # target is (N, 1)
                y = batch.y.view(-1, 1)
                # take argmax over classes to get the prediction
                pred = torch.argmax(pred, dim=1).view(-1, 1)
            else:
                # target is same shape as pred
                y = batch.y.view(pred.shape)
                # pred is unchanged

            y_true.append(y.detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)

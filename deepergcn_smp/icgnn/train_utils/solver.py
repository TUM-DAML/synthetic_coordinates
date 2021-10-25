# flush print statements immediately to stdout so that we can see them
# in the slurm output
import functools
print = functools.partial(print, flush=True)

from collections import defaultdict
import time

import torch
import torch.nn.functional as F

import seaborn as sns
import numpy as np
from tqdm import tqdm

from ..models.misc import count_parameters
from ..data_utils.data import set_train_val_test_split

def solve(model, graph, seeds, max_epochs, display_step, patience, learning_rate, l2_reg, use_amsgrad=False, linegraph=None):
    print(f'Num parameters: {count_parameters(model)}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running on:\t\t', device)

    graph = graph.to(device)
    best_dict = defaultdict(list)

    start_time = time.perf_counter()

    for seed_ndx, seed in enumerate(seeds):
        print(f'Seed: {seed_ndx + 1}/{len(seeds)}')
        graph = set_train_val_test_split(
            seed,
            graph,
            num_development=1500
        ).to(device)

        print('Training on graph:', graph)
        print('Training on linegraph:', linegraph)
        print('Train nodes: \t', torch.sum(graph.train_mask).item())
        print('Val nodes: \t', torch.sum(graph.val_mask).item())
        print('Test nodes: \t', torch.sum(graph.test_mask).item())

        # send model to device before giving the optimizer its parameters
        model.to(device).reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                    weight_decay=l2_reg, amsgrad=use_amsgrad)

        patience_counter = 0
        tmp_dict = {'val_acc': 0}

        train_loss_hist, val_loss_hist = [], []
        train_acc_hist, val_acc_hist = [], []

        for epoch in range(max_epochs):
            if patience_counter == patience:
                break

            train(model, optimizer, graph, linegraph=linegraph)
            losses, accs = evaluate(model, graph, linegraph=linegraph)

            train_loss, val_loss = losses
            train_acc, val_acc = accs

            train_loss_hist.append(train_loss)
            val_loss_hist.append(val_loss)
            train_acc_hist.append(train_acc)
            val_acc_hist.append(val_acc)

            if val_acc < tmp_dict['val_acc']:
                patience_counter += 1
            else:
                patience_counter = 0
                tmp_dict['epoch'] = epoch
                tmp_dict['val_acc'] = val_acc
                tmp_dict['train_acc'] = train_acc

            if epoch % display_step == 0:
                print(f"Epoch: {epoch:03d}, \t T.acc: {train_acc:.4f}, V.acc: {val_acc:.4f}               T.loss: {train_loss:.4f}, V.loss: {val_loss:.4f}")

        for k, v in tmp_dict.items():
            best_dict[k].append(v)
        print('Best till now:', best_dict)

    best_dict['duration'] = time.perf_counter() - start_time

    # summarize results
    # calculate mean accuracy and confidence interval
    boots_series = sns.algorithms.bootstrap(best_dict['val_acc'], func=np.mean,
                                            n_boot=1000)
    best_dict['val_acc_ci'] = np.max(np.abs(sns.utils.ci(boots_series, 95)
                                    - np.mean(best_dict['val_acc'])))
    for k, v in best_dict.items():
        if 'acc_ci' not in k and k != 'duration':
            best_dict[k] = np.mean(best_dict[k])

    return dict(best_dict)

def solve_graphcls(model, train_loader, val_loader,
                                max_epochs, display_step, patience,
                                learning_rate, l2_reg, use_amsgrad,
                                binary_cls=False,
                                ogb_evaluator=None):
    print(f'Num parameters: {count_parameters(model)}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running on:\t\t', device)

    start_time = time.perf_counter()

    # send model to device before giving the optimizer its parameters
    model.to(device).reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                weight_decay=l2_reg, amsgrad=use_amsgrad)

    patience_counter = 0
    best_val_acc = 0
    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []

    for epoch in range(max_epochs):
        if patience_counter == patience:
            break

        train_loss, train_acc = step_graphcls(model, train_loader, optimizer,
                                                is_train=True,
                                                binary_cls=binary_cls,
                                                ogb_evaluator=ogb_evaluator)
        with torch.no_grad():
            val_loss, val_acc = step_graphcls(model, val_loader, optimizer=None,
                                                    is_train=False,
                                                    binary_cls=binary_cls,
                                                    ogb_evaluator=ogb_evaluator)

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

        if val_acc < best_val_acc:
            patience_counter += 1
        else:
            best_val_acc = val_acc
            patience_counter = 0

        # if epoch % display_step == 0:
        print(f"Epoch: {epoch:03d}, \t T.acc: {train_acc:.4f}, V.acc: {val_acc:.4f} T.loss: {train_loss:.4f}, V.loss: {val_loss:.4f}")

    result = {}
    result['val_acc'] = max(val_acc_hist)

    result['train_loss_hist'] = train_loss_hist
    result['val_loss_hist'] = val_loss_hist

    result['train_acc_hist'] = train_acc_hist
    result['val_acc_hist'] = val_acc_hist

    result['duration'] = time.perf_counter() - start_time

    return dict(result)

def step_graphcls(model, loader, optimizer=None, is_train=False,
                   binary_cls=False, ogb_evaluator=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    y_true, y_pred = [], []

    if is_train:
        model.train()
    else:
        model.eval()

    loss_all, correct = 0, 0

    for batch in tqdm(loader, desc='batch'):
        batch = batch.to(device)

        if is_train:
            optimizer.zero_grad()
        output = model(batch=batch)

        pred = output.max(dim=1)[1]
        correct += pred.eq(batch.y).sum().item()

        # compute loss only over labeled samples
        is_labeled = (batch.y == batch.y)

        if binary_cls:
            # OGB - remove unlabeled samples
            # TUdataset: change [n,] to [n, 1]
            preds = output.to(torch.float32)[is_labeled]
            target = batch.y.to(torch.float32)[is_labeled].reshape(preds.shape)

            loss = F.binary_cross_entropy_with_logits(preds, target)
        else:
            loss = F.nll_loss(output, batch.y)

        if is_train:
            try:
                loss.backward()
                optimizer.step()
            except RuntimeError:
                print('Cant run backward pass! Skipping batch')
                return 0, 0

        loss_all += len(batch) * loss.item()

        if ogb_evaluator:
            y_true.append(batch.y.view(output.shape).detach().cpu())
            y_pred.append(output.detach().cpu())

    if ogb_evaluator:
        y_true = torch.cat(y_true, dim = 0).numpy()
        y_pred = torch.cat(y_pred, dim = 0).numpy()
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        result = ogb_evaluator.eval(input_dict)
        metric = list(result.values())[0]

    else:
        metric = correct / len(loader.dataset)

    return loss_all / len(loader.dataset), metric

# single training iteration on a masked part of the graph
def train(model, optimizer, graph, linegraph=None):
    '''
    Optimize the model
    model: nn.Module
    optimizer: torch.optim.X
    graph: torch_geometric.data
    linegraph: torch_geometric.data (optional) linegraph of graph
    '''
    model.train()
    optimizer.zero_grad()
    # classification loss
    if linegraph:
        out = model(graph, linegraph)
    else:
        out = model(graph.x, graph.edge_index)
    loss = F.nll_loss(out[graph.train_mask], graph.y[graph.train_mask])
    loss.backward()
    optimizer.step()

def evaluate(model, graph, masks=('train_mask', 'val_mask'), linegraph=None):
    '''
    Evalute on the train and val sets

    model: nn.Module
    graph: torch_geometric.data
    masks: list of masks to run on. options: (train_mask, val_mask, test_mask)
    '''
    losses, accs = [], []

    with torch.no_grad():
        model.eval()
        if linegraph:
            logits = model(graph, linegraph)
        else:
            logits = model(graph.x, graph.edge_index)

        for _, mask in graph(*masks):
            pred = logits[mask].max(1)[1]

            loss = F.nll_loss(logits[mask], graph.y[mask])
            losses.append(loss.detach().cpu().numpy())

            acc = pred.eq(graph.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)

    return losses, accs

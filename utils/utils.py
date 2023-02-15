import torch
import pickle
import random
import glob
import numpy as np
from collections import defaultdict

def any2device(value, device):
    """Transfer object value to given device with all internal content
        :param value: object which need to transfer
               device: 'cpu' or 'gpu' machine for transfer
        :return: torch.Tensor
    """
    if torch.is_tensor(value):
        return value.to(device)
    elif isinstance(value, dict):
        return {k: any2device(v, device) for k, v in value.items()}
    elif isinstance(value, list) or isinstance(value, tuple):
        return [any2device(subvalue, device) for subvalue in value]

    assert "Your object type is not implemented"


def fit_model(model, dataloaders, criterion, optim, args, output_dir):
    """Train and eval loops for given params"""
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device("cuda:" + str(args.device))
    else:
        device = 'cpu'
    best_loss = 100
    criterion.to(device)
    model.to(device)
    for epoch in range(args.num_epochs):
        run_loss = train_epoch(model, dataloaders['train'], criterion, optim, device)
        if run_loss < best_loss:
            best_loss = run_loss
            torch.save(model.to(torch.device("cpu")), output_dir + 'best.pkl')
            model.to(device)

        eval_results = eval_model(model, dataloaders['valid'], criterion, device, 'valid', str(epoch))

    model = torch.load(output_dir + 'best.pkl')
    for name in dataloaders.keys():
        eval_results = eval_model(model, dataloaders[name], criterion, device, name, "after")
        with open(output_dir + name + '_results.pkl', 'wb') as file:
            pickle.dump(eval_results, file)


def train_epoch(model, dataloader, criterion, optim, device):
    """Train single epoch"""
    model.train()
    run_loss = 0
    for batch in dataloader:
        batch = any2device(batch, device)
        output = model(batch)
        loss = criterion(output, batch)

        optim.zero_grad()
        loss.backward()
        optim.step()

        run_loss += loss.item()

    return run_loss / len(dataloader)


def eval_model(model, dataloader, criterion, device, name, epoch):
    """Evaluate and print metrics for validation dataset"""
    model.eval()
    run_loss = 0
    sub_preds = []
    sub_labels = []
    for batch in dataloader:
        batch = any2device(batch, device)
        with torch.no_grad():
            output = model(batch)
            loss = criterion(output, batch)
            run_loss += loss.item()
            sub_preds.append(output['log_preds'].detach().cpu())
            sub_labels.append(batch['labels'].numpy().reshape(-1,1))
    sub_preds = np.vstack(sub_preds)
    sub_labels = np.vstack(sub_labels).reshape((-1))
    acc = np.sum(np.argmax(sub_preds, axis=-1) == sub_labels) / sub_preds.shape[0]
    F1 = f1_metrics(sub_preds, sub_labels)

    ans = {'loss': run_loss / len(dataloader),
           'acc': acc,
           **F1
           }
    print(epoch, " ", name, "  loss  ", ans['loss'], "  acc  ", ans['acc'],
          " macro F1  ", ans['F1_macro'], " micro F1  ", ans['F1_micro']
          )

    return ans


def infer_model(model, dataloader, device, key):
    """Run model without learning"""
    model.eval()
    ans = []
    for batch in dataloader:
        batch = any2device(batch, device)
        with torch.no_grad():
            # model = torch.nn.Sequential(*(list(model.children())[:-2]))
            output = model(batch)
            ans.append(output[key].detach().cpu())
    ans = np.vstack(ans)

    return ans


def test_model(name, path):
    """Package test results from experiments into one map"""
    path = path + 'exp_{}_*/'.format(name)
    paths = glob.glob(path)
    result = defaultdict(list)
    for path in paths:
        with open(path, 'rb') as file:
            res = pickle.load(file)
        for k in res:
            result[k].append(res[k])
    return result


def f1_metrics(sub_preds, labels, beta=1):
    """Calculate F1-metric with macro and micro averaging"""
    f1_macro = []
    tp, fp, fn = 0, 0, 0
    for label in np.unique(labels):
        TruePos = np.sum((np.argmax(sub_preds, axis=-1) == label) * (labels == label))
        FalsePos = np.sum((np.argmax(sub_preds, axis=-1) == label) * (labels != label))
        FalseNeg = np.sum((np.argmax(sub_preds, axis=-1) != label) * (labels == label))
        tp, fp, fn = tp + TruePos, fp + FalsePos, fn + FalseNeg
        f1 = TruePos / (TruePos + 0.5 * FalsePos * FalseNeg) if (TruePos + 0.5 * FalsePos * FalseNeg) > 0 else 0.0
        f1_macro.append(f1)

    return {
        'F1_macro': np.mean(f1_macro),
        'F1_micro': tp / (tp + 0.5 * fp * fn)
    }

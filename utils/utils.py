import torch
import numpy as np


def any2device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    elif isinstance(value, dict):
        return {k: any2device(v, device) for k, v in value.items()}
    elif isinstance(value, list) or isinstance(value, tuple):
        return [any2device(subvalue, device) for subvalue in value]

    assert "Your object type is not implemented"


def fit_model(model, dataloaders, criterion, optim, args, output_dir):
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


def train_epoch(model, dataloader, criterion, optim, device):
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

    ans = {'loss': run_loss / len(dataloader),
           'acc': acc
           }
    print(epoch, " ", name, "  eval loss  ", ans['loss'], "  eval acc  ", ans['acc'])

    return ans


def infer_model(model, dataloader, device, key):
    model.eval()
    ans = []
    for batch in dataloader:
        batch = any2device(batch, device)
        with torch.no_grad():
            output = model(batch)
            ans.append(output[key].detach().cpu())
    ans = np.vstack(ans)

    return ans
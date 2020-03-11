import torch
import torch.nn as nn
from transformers import AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def train_function(data, model, opt_param, lr):
    optimizer = get_optimizer(opt_param, lr)
    model.train()

    for idx, value in tqdm(enumerate(data), total=len(data)):
        optimizer.zero_grad()

        ids = value[0]
        mask = value[1]
        token_ids = value[2]
        targets = value[3].view(-1, 1)

        outputs = model(input_ids=ids, attention_mask=mask, token_type_ids=token_ids)
        loss = loss_func(pred=outputs, true=targets)
        loss.backward()

        optimizer.step()

    print(f"Loss {loss}")


def eval_function(data, model):
    model.eval()
    true = []
    pred = []
    with torch.no_grad():
        for idx, value in tqdm(enumerate(data), total=len(data)):

            ids = value[0]
            mask = value[1]
            token_ids = value[2]
            targets = value[3].view(-1, 1)

            outputs = model(input_ids=ids, attention_mask=mask, token_type_ids=token_ids)
            predicts = torch.gt(outputs, 0.5).double()

            true.extend(targets.squeeze().tolist())
            pred.extend(predicts.squeeze().tolist())

    acc = accuracy_score(y_true=true, y_pred=pred)
    print(f"Accuracy : {acc}")


def loss_func(pred, true):
    return nn.BCEWithLogitsLoss()(pred, true)


def get_optimizer(opt_param, lr):
    return AdamW(opt_param, lr=lr)

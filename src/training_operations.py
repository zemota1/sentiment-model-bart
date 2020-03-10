import numpy as np
import torch
import torch.nn as nn
from transformers import AdamW
from tqdm import tqdm


def train_function(data, model, lr):
    optimizer = get_optimizer(model, lr)
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
    true_labels = []
    pred_labels = []
    with torch.no_grad:
        for idx, value in enumerate(data):

            ids = value[0]
            mask = value[1]
            token_ids = value[2]
            targets = value[3].view(-1, 1)

            outputs = model(input_ids=ids, attention_mask=mask, token_type_ids=token_ids)

            true_labels.append(targets)
            pred_labels.append(outputs)

        true_labels = np.array(true_labels)

        pred_labels = np.array(pred_labels)
        pred_labels[pred_labels >= 0.5] = 1
        pred_labels[pred_labels < 0.5] = 0
        acc = sum(1 for x, y in zip(true_labels, pred_labels) if x == y) / len(true_labels)

    print(f"Accuracy: {acc}")
    return acc


def loss_func(pred, true):
    return nn.BCEWithLogitsLoss()(pred, true)


def get_optimizer(model, lr):
    return AdamW(model.parameters(), lr=lr)

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.dataset import IMDBDataset
from src.model import BERTModel
from src.training_operations import train_function, eval_function

RANDOM_STATE = 42
DATA_PATH = "../data/imdb-dataset.csv"
NUMBER_EPOCHS = 10


def pre_process(path):
    df = pd.read_csv(path)
    df = df.fillna('')
    df.sentiment = pd.factorize(df.sentiment)[0]

    train, test = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df.sentiment.values
    )

    return train, test


def get_loaders(train_set, test_set):

    train_dataset = IMDBDataset(train_set.review, train_set.sentiment)
    test_dataset = IMDBDataset(test_set.review, test_set.sentiment)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=False
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=4,
        shuffle=False
    )

    return train_loader, test_loader


def main():

    train_set, test_set = pre_process(path=DATA_PATH)
    train_data_loader, test_data_loader = get_loaders(train_set, test_set)

    model = BERTModel()
    for epoch in range(NUMBER_EPOCHS):
        train_function(data=train_data_loader, model=model, lr=1e-3)
        eval_function(data=test_data_loader)


if __name__ == '__main__':
    main()

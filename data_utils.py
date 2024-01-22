import pandas as pd


def get_split(csv_path, test_room, split):
    df = pd.read_csv(csv_path)

    test_df = df[df["path"].str.contains("|".join(test_room))]
    train_df = df[~df["path"].str.contains("|".join(test_room))]

    if split == "train":
        return train_df
    elif split == "test":
        return test_df

import argparse
import os

import pandas as pd


def load_and_merge_csvs(input_dir):
    return pd.concat(
        [
            pd.read_csv(os.path.join(input_dir, f))
            for f in os.listdir(input_dir)
            if f.endswith(".csv")
        ],
        ignore_index=True,
    ).dropna()


def split_dataset(df):
    train = df.sample(frac=0.7, random_state=42)
    remaining = df.drop(train.index)
    val = remaining.sample(frac=0.5, random_state=42)
    test = remaining.drop(val.index)
    return train, val, test


def save_splits(train, val, test, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(f"{output_dir}/train.csv", index=False)
    val.to_csv(f"{output_dir}/val.csv", index=False)
    test.to_csv(f"{output_dir}/test.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", required=True)
    args = parser.parse_args()

    df = load_and_merge_csvs(args.input_data)
    train, val, test = split_dataset(df)
    save_splits(train, val, test)


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import json
import argparse


def main(args):
    df = pd.read_csv(args.original_file)
    df["id"] = df["id"].astype(str)
    with open(args.rewritten_file) as f:
        rewritten = json.load(f)
    columns = [
        "subject_id",
        "id",
        "Care",
        "Fairness",
        "Loyalty",
        "Authority",
        "Purity",
        "age",
        "gender",
        "z.care",
        "z.fairness",
        "z.loyalty",
        "z.authority",
        "z.purity",
        "c.care",
        "c.fairness",
        "c.loyalty",
        "c.authority",
        "c.purity"
    ]
    rewritten_df = pd.DataFrame({
        "id": rewritten.keys(),
        "text": rewritten.values()
    })

    rewritten_df = pd.merge(
        rewritten_df,
        df[columns],
        on="id",
        how="left"
    )

    rewritten_df["id"] = rewritten_df["id"].astype(int)

    # create a new dataframe with the text being the concatenated text for the sorted ids per subject_id
    new_df = rewritten_df.groupby("subject_id").apply(
        lambda x: x.sort_values("id")["text"].str.cat(sep="; ")
    ).reset_index(name="text")
    new_df = pd.merge(
        new_df,
        rewritten_df[[col for col in columns if col != "id"]].drop_duplicates(),
        on="subject_id",
        how="left"
    )

    new_df.to_csv(args.rewritten_file.replace(".json", ".csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_file", type=str, required=True)
    parser.add_argument("--rewritten_file", type=str, required=True)

    args = parser.parse_args()
    main(args)

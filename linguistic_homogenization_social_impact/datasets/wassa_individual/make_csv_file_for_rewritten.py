import pandas as pd
import numpy as np
import json
import argparse
from IPython import embed


def main(args):
    df = pd.read_csv(args.original_file)
    df["id"] = df["id"].astype(str)
    with open(args.rewritten_file) as f:
        rewritten = json.load(f)
    columns = [
        "id",
        "iri.perspective",
        "c.iri.perspective",
        "z.iri.perspective",
        "iri.distress",
        "c.iri.distress",
        "z.iri.distress",
        "iri.fantasy",
        "c.iri.fantasy",
        "z.iri.fantasy",
        "iri.concern",
        "c.iri.concern",
        "z.iri.concern",
        "speaker_id",
        "article_id",
        "gender",
        "age",
        "race",
        "education",
        "income",
        "O",
        "C",
        "E",
        "A",
        "N",
        "empathy",
        "cempathy",
        "zempathy",
        "distress",
        "cdistress",
        "zdistress",
        "emotion_1",
    ]
    rewritten_df = pd.DataFrame({"id": rewritten.keys(), "text": rewritten.values()})
    print(rewritten_df.shape)
    rewritten_df = pd.merge(rewritten_df, df[columns], on="id", how="left")
    print(rewritten_df.shape)

    rewritten_df.to_csv(args.rewritten_file.replace(".json", ".csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_file", type=str, required=True)
    parser.add_argument("--rewritten_file", type=str, required=True)

    args = parser.parse_args()
    main(args)

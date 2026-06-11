import pandas as pd
import numpy as np
import json
import argparse
import os
from collections import defaultdict
from tqdm import tqdm
from IPython import embed


def add_mrc_features(input_file, text_column):
    pass


def add_nrc_features(input_file, text_column):
    EMOTION_CATEGORIES = [
        "anger",
        "anticipation",
        "disgust",
        "fear",
        "joy",
        "negative",
        "positive",
        "sadness",
        "surprise",
        "trust",
    ]
    emotion_dictionaries = defaultdict(list)
    parent_folder = "../dictionaries/NRC-Emotion-Lexicon/OneFilePerEmotion"
    for emotion in tqdm(EMOTION_CATEGORIES, leave=False):
        with open(
            os.path.join(parent_folder, f"{emotion}-NRC-Emotion-Lexicon.txt")
        ) as f:
            lines = f.readlines()
        for line in lines:
            word, label = line.split("\t")
            if int(label) == 1:
                emotion_dictionaries[emotion].append(word.lower())

    df = pd.read_csv(input_file)
    results_dict = defaultdict(list)
    for _, row in tqdm(df.iterrows(), leave=False, total=len(df)):
        try:
            text = row[text_column].lower()
            text_length = len(text.split())
            for emotion in EMOTION_CATEGORIES:
                count = 0
                for word in emotion_dictionaries[emotion]:
                    count += text.count(word)
                results_dict[emotion].append(count / text_length)
        except Exception as e:
            print(e)
            for emotion in EMOTION_CATEGORIES:
                results_dict[emotion].append(np.nan)

    for emotion in EMOTION_CATEGORIES:
        df[f"nrc.{emotion}"] = results_dict[emotion]
    df.to_csv(input_file, index=False)


def add_mfd_features(input_file, text_column):
    MFD_CATEGORIES_DICT = {
        1: "care.virtue",
        2: "care.vice",
        3: "fairness.virtue",
        4: "fairness.vice",
        5: "loyalty.virtue",
        6: "loyalty.vice",
        7: "authority.virtue",
        8: "authority.vice",
        9: "sanctity.virtue",
        10: "sanctity.vice",
    }
    MFD_CATEGORIES = MFD_CATEGORIES_DICT.values()
    with open("../dictionaries/MFD2/mfd2.0.dic") as f:
        lines = f.readlines()
    num_of_percent_signs = 0
    mfd_dict = defaultdict(list)

    for index, line in enumerate(lines):
        if line.strip() == "%":
            num_of_percent_signs += 1
        if num_of_percent_signs < 2:
            continue
        if num_of_percent_signs == 2:
            num_of_percent_signs += 1
            continue

        word, category = line.split("\t")
        mfd_dict[MFD_CATEGORIES_DICT[int(category)]].append(word.lower())

    df = pd.read_csv(input_file)
    mfd_features_dict = defaultdict(list)
    for _, row in tqdm(df.iterrows(), leave=False, total=len(df)):
        try:
            text = row[text_column].lower()
            text_length = len(text.split())
            for category in MFD_CATEGORIES:
                count = 0
                for word in mfd_dict[category]:
                    count += text.count(word)
                mfd_features_dict[category].append(count / text_length)
        except Exception as e:
            print(e)
            for category in MFD_CATEGORIES:
                mfd_features_dict[category].append(np.nan)

    for category in MFD_CATEGORIES:
        df[f"mfd.{category}"] = mfd_features_dict[category]
    df.to_csv(input_file, index=False)


def add_empathy_features(input_file, text_column):
    empathy_categories = [
        "low_empathy",
        "high_empathy",
        "low_distress",
        "high_distress",
    ]
    empathy_df = pd.read_csv("../dictionaries/empathy/empathy_lexicon.txt")
    empathy_median = empathy_df["rating"].median()
    distress_df = pd.read_csv("../dictionaries/distress/distress_lexicon.txt")
    distress_median = distress_df["rating"].median()
    empathy_dictionaries = defaultdict(list)
    empathy_dictionaries["high_empathy"].extend(
        [
            word.lower()
            for word in empathy_df[empathy_df["rating"] >= empathy_median][
                "word"
            ].tolist()
        ]
    )
    empathy_dictionaries["low_empathy"].extend(
        [
            word.lower()
            for word in empathy_df[empathy_df["rating"] < empathy_median][
                "word"
            ].tolist()
        ]
    )
    empathy_dictionaries["high_distress"].extend(
        [
            word.lower()
            for word in distress_df[distress_df["rating"] >= distress_median][
                "word"
            ].tolist()
        ]
    )

    empathy_dictionaries["low_distress"].extend(
        [
            word.lower()
            for word in distress_df[distress_df["rating"] < distress_median][
                "word"
            ].tolist()
        ]
    )

    df = pd.read_csv(input_file)
    results_dict = defaultdict(list)
    for _, row in tqdm(df.iterrows(), leave=False, total=len(df)):
        try:
            text = row[text_column].lower()
            text_length = len(text.split())
            for category in empathy_categories:
                count = 0
                for word in empathy_dictionaries[category]:
                    count += text.count(word)
                results_dict[category].append(count / text_length)
        except Exception as e:
            print(e)
            for category in empathy_categories:
                results_dict[category].append(np.nan)
    for category in empathy_categories:
        df[f"empathy.{category}"] = results_dict[category]

    df.to_csv(input_file, index=False)


def main(args):
    for input_file in [
        file for file in os.listdir(args.input_folder) if file.endswith(".csv")
    ]:
        if args.dictionary_type == "nrc":
            add_nrc_features(
                input_file=os.path.join(args.input_folder, input_file),
                text_column=args.text_column,
            )
        elif args.dictionary_type == "mfd":
            add_mfd_features(
                input_file=os.path.join(args.input_folder, input_file),
                text_column=args.text_column,
            )
        elif args.dictionary_type == "empathy":
            add_empathy_features(
                input_file=os.path.join(args.input_folder, input_file),
                text_column=args.text_column,
            )
        # elif args.dictionary_type == "mrc":
        #     add_mrc_features(
        #         input_file=os.path.join(args.input_folder, input_file),
        #         text_column=args.text_column,
        #     )
        else:
            raise ValueError("Invalid dictionary type")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--text_column", type=str, required=False, default="text")
    parser.add_argument("--dictionary_type", type=str, required=True)

    args = parser.parse_args()
    main(args)

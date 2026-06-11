import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier
from IPython import embed
import logging

# from scipy.stats import pearsonr
import random
from datetime import datetime
import os

import warnings

warnings.filterwarnings("ignore")
random_seed = np.random.randint(0, 100000000)

POLITICAL_LABEL_TO_INDEX = {
    "over 70": 0,
    "41-55": 1,
    "27-40": 2,
    "56-70": 3,
    "R": 0,
    "D": 1,
    "M": 0,
    "F": 1,
}

ordered_age_group_indices = {
    "27-40": 0,
    "41-55": 1,
    "56-70": 2,
    "over 70": 3,
}

from_current_to_ordered_age_group_indices = {
    0: 3,
    1: 1,
    2: 0,
    3: 2,
}


def create_log_file(args):
    args_dict = vars(args)
    log_file_name = "_".join(
        [
            f"{key}={value}"
            for key, value in args_dict.items()
            if key
            in [
                "features",
                "label_column",
                "model_type",
                "output_type",
                "reverse",
                "rewritten_type",
                "LLM_model",
            ]
        ]
    )
    log_file_name += f"_random_seed={random_seed}"
    if not os.path.exists(f"{args.dataset}_logs"):
        os.makedirs(f"{args.dataset}_logs")
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f"{args.dataset}_logs/{log_file_name}.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
    return log_file_name


def get_tfidf_features(df, rewritten_df, args):
    USE_IDF = True
    TFIDF_MIN = 0.1
    TFIDF_MAX = 0.9
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=TFIDF_MIN, max_df=TFIDF_MAX, use_idf=USE_IDF
    )
    vectorizer.fit(df[args.text_column].values)
    scaler = StandardScaler()
    features = vectorizer.transform(df[args.text_column].values)
    features = scaler.fit_transform(features.toarray())
    ids_in_embeddings = df[args.id_column].values.tolist()

    features_rewritten = vectorizer.transform(rewritten_df[args.text_column].values)
    features_rewritten = scaler.transform(features_rewritten.toarray())
    rewritten_ids_in_embeddings = rewritten_df[args.id_column].values.tolist()

    return features, features_rewritten, ids_in_embeddings, rewritten_ids_in_embeddings


def get_openai_features(df, rewritten_df, embeddings, rewritten_embeddings, args):
    ids_in_embeddings = [
        key for key in df[args.id_column].values if key in embeddings.keys()
    ]
    features = np.array([embeddings[key] for key in ids_in_embeddings])
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    rewritten_ids_in_embeddings = [
        key
        for key in rewritten_df[args.id_column].values
        if key in rewritten_embeddings.keys()
    ]
    features_rewritten = np.array(
        [rewritten_embeddings[key] for key in rewritten_ids_in_embeddings]
    )
    features_rewritten = scaler.transform(features_rewritten)

    return features, features_rewritten, ids_in_embeddings, rewritten_ids_in_embeddings


def get_classification_labels(
    df, rewritten_df, train_ids_in_df, test_ids_in_df, ids_in_rewritten_df, args
):
    def find_labels(df, ids, args):
        if args.dataset != "political":
            labels_dictionary = dict(
                zip(df[args.id_column].values, df[f"c{args.label_column}"].values)
            )
            labels = np.array([labels_dictionary[key] for key in ids])
            labels = np.where(labels == "y", 1, 0)
        elif args.dataset == "political":
            labels_dictionary = dict(
                zip(df[args.id_column].values, df[args.label_column].values)
            )
            labels = np.array([labels_dictionary[key] for key in ids])
            labels = np.array([POLITICAL_LABEL_TO_INDEX[label] for label in labels])
        else:
            raise ValueError("Invalid dataset")
        return labels

    train_labels = find_labels(df, train_ids_in_df, args)
    test_labels = find_labels(df, test_ids_in_df, args)
    labels_rewritten = find_labels(rewritten_df, ids_in_rewritten_df, args)

    return train_labels, test_labels, labels_rewritten


def get_svm_model(args):
    parameters = {"kernel": ["rbf"], "C": [0.1, 1, 10]}
    model = SVC(probability=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    clf = GridSearchCV(
        model,
        parameters,
        cv=cv,
        verbose=2,
        scoring=("f1" if args.dataset != "political" else "f1_macro"),
    )
    # change the above to f1 with weighted average
    return clf


def get_gradient_boosting_model(args):
    parameters = {
        "n_estimators": [16, 32],
        "learning_rate": [0.1, 0.01],
        "max_depth": [3, 5],
    }
    model = GradientBoostingClassifier()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    clf = GridSearchCV(
        model,
        parameters,
        cv=cv,
        verbose=2,
        scoring=("f1" if args.dataset != "political" else "f1_macro"),
    )
    return clf


def get_random_forest_model(args):
    n_estimators = [int(x) for x in np.linspace(start=200, stop=500, num=2)]
    max_depth = [int(x) for x in np.linspace(10, 40, num=2)]
    min_samples_split = [4]
    min_samples_leaf = [4]
    parameters = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
    }
    model = RandomForestClassifier()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    clf = GridSearchCV(
        model,
        parameters,
        cv=cv,
        verbose=2,
        scoring=("f1" if args.dataset != "political" else "f1_macro"),
    )

    return clf


def get_regression_model(args):

    parameters = {
        "C": [0.1, 1, 10],
    }
    model = LogisticRegression()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    clf = GridSearchCV(
        model,
        parameters,
        cv=cv,
        verbose=1,
        scoring=("f1" if args.dataset != "political" else "f1_macro"),
    )

    return clf


def load_files(args):
    df = pd.read_csv(args.input_file)
    df[args.id_column] = df[args.id_column].astype(str)

    rewritten_df = pd.read_csv(args.rewritten_input_file)
    rewritten_df = rewritten_df.dropna(subset=[args.text_column])
    rewritten_df[args.id_column] = rewritten_df[args.id_column].astype(str)

    with open(args.input_file_embeddings, "rb") as f:
        embeddings = json.load(f)
        embeddings = {
            key: data["data"][0]["embedding"] for key, data in embeddings.items()
        }

    with open(args.rewritten_input_file_embeddings, "rb") as f:
        rewritten_embeddings = json.load(f)
        rewritten_embeddings = {
            key: data["data"][0]["embedding"]
            for key, data in rewritten_embeddings.items()
        }

    if args.reverse == 1:
        df, rewritten_df = rewritten_df, df
        embeddings, rewritten_embeddings = rewritten_embeddings, embeddings

    return df, rewritten_df, embeddings, rewritten_embeddings


def main(args):
    log_file_name = create_log_file(args)

    df, rewritten_df, embeddings, rewritten_embeddings = load_files(args)

    # get the features
    if args.features == "tfidf":
        (
            features,
            features_rewritten,
            ids_in_df,
            ids_in_rewritten_df,
        ) = get_tfidf_features(df, rewritten_df, args)
    elif args.features == "openai":
        (
            features,
            features_rewritten,
            ids_in_df,
            ids_in_rewritten_df,
        ) = get_openai_features(
            df, rewritten_df, embeddings, rewritten_embeddings, args
        )
    else:
        raise ValueError("Invalid feature type")

    train_features, test_features, train_ids_in_df, test_ids_in_df = train_test_split(
        features, ids_in_df, test_size=0.2, random_state=random_seed
    )
    updated_ids_in_rewritten_df = []
    updated_features_rewritten = []
    for feature, id in zip(features_rewritten, ids_in_rewritten_df):
        if id in test_ids_in_df:
            updated_ids_in_rewritten_df.append(id)
            updated_features_rewritten.append(feature)
    print("number of test ids in rewritten df", len(updated_ids_in_rewritten_df))

    ids_in_rewritten_df = np.array(updated_ids_in_rewritten_df)
    features_rewritten = np.array(updated_features_rewritten)

    train_labels, test_labels, labels_rewritten = get_classification_labels(
        df, rewritten_df, train_ids_in_df, test_ids_in_df, ids_in_rewritten_df, args
    )

    # write the clf with parameter fine-tuning
    if args.model_type == "svm":
        clf = get_svm_model(args)
    elif args.model_type == "gradient_boosting":
        clf = get_gradient_boosting_model(args)
    elif args.model_type == "random_forest":
        clf = get_random_forest_model(args)
    elif args.model_type == "regression":
        clf = get_regression_model(args)
    else:
        raise ValueError("Invalid model type")

    clf.fit(train_features, train_labels)

    # print the results of the best model
    logging.info("best score:")
    logging.info(clf.best_score_)
    logging.info("best score on test essays")
    logging.info(clf.score(test_features, test_labels))
    logging.info("complete report")
    extra_data = {}
    # put the test ids, with labels, and confidence scores
    extra_data["test"] = {
        "ids": (
            test_ids_in_df if type(test_ids_in_df) == list else test_ids_in_df.tolist()
        ),
        "labels": test_labels.tolist(),
        "predicted_labels": clf.predict(test_features).tolist(),
        "confidence_scores": clf.predict_proba(test_features).tolist(),
    }

    logging.info("-----REWRITTEN-----")
    logging.info("best score on rewritten essays")
    logging.info(clf.score(features_rewritten, labels_rewritten))
    extra_data["rewritten"] = {
        "ids": ids_in_rewritten_df.tolist(),
        "labels": labels_rewritten.tolist(),
        "predicted_labels": clf.predict(features_rewritten).tolist(),
        "confidence_scores": clf.predict_proba(features_rewritten).tolist(),
    }

    with open(f"{args.dataset}_logs/{log_file_name}.json", "w") as f:
        json.dump(extra_data, f)

    logging.info("-" * 20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--LLM_model", type=str, required=True)

    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the CSV input file"
    )

    parser.add_argument(
        "--input_file_embeddings",
        type=str,
        required=False,
        help="Path to the json file that has the embeddings for the input file",
    )

    parser.add_argument(
        "--rewritten_input_file",
        type=str,
        required=False,
        help="Path to the CSV rewritten input file that has the same columns as the as the input file",
    )
    parser.add_argument(
        "--rewritten_input_file_embeddings",
        type=str,
        required=False,
        help="Path to the json file that has the embeddings for the rewritten input file",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["essays", "wassa", "facebook", "political"],
    )
    parser.add_argument(
        "--text_column",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--id_column",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--label_column",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["svm", "random_forest", "regression", "gradient_boosting"],
    )

    parser.add_argument(
        "--features", type=str, required=True, choices=["tfidf", "openai"]
    )

    parser.add_argument(
        "--output_type",
        type=str,
        required=True,
        choices=["classification", "regression"],
    )

    parser.add_argument("--reverse", type=int, default=0)

    parser.add_argument(
        "--rewritten_type", type=str, required=False, default="syntax_grammar"
    )

    args = parser.parse_args()

    main(args)

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
from datasets import load_metric
import argparse
import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
import json
from IPython import embed
import logging
from datasets import Dataset, DatasetDict

import random
from datetime import datetime
import os

import warnings

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

dataset_to_max_length = {
    "essays": 1500,
    "wassa": 4096,
    "facebook": 4096,
    "political": 4096,
}

warnings.filterwarnings("ignore")
random_seed = np.random.randint(0, 100000000)


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


def get_all_splits(args):
    df = pd.read_csv(args.input_file)
    df[args.id_column] = df[args.id_column].astype(str)

    rewritten_df = pd.read_csv(args.rewritten_input_file)
    rewritten_df = rewritten_df.dropna(subset=[args.text_column])
    rewritten_df[args.id_column] = rewritten_df[args.id_column].astype(str)

    if args.reverse == 1:
        df, rewritten_df = rewritten_df, df

    all_ids = set(df[args.id_column].values)

    np.random.seed(random_seed)
    train_val_ids = np.random.choice(
        list(all_ids), int(0.8 * len(all_ids)), replace=False
    )
    np.random.seed(random_seed)
    train_ids = np.random.choice(
        train_val_ids, int(0.8 * len(train_val_ids)), replace=False
    )
    val_ids = list(set(train_val_ids) - set(train_ids))

    test_ids = list(all_ids - set(train_val_ids))

    rewritten_ids = set(rewritten_df[args.id_column].values)
    ids_in_common_with_test_and_rewritten = list(
        rewritten_ids.intersection(set(test_ids))
    )

    train_df = df[df[args.id_column].isin(train_ids)]
    val_df = df[df[args.id_column].isin(val_ids)]
    test_df = df[df[args.id_column].isin(ids_in_common_with_test_and_rewritten)]
    rewritten_df = rewritten_df[
        rewritten_df[args.id_column].isin(ids_in_common_with_test_and_rewritten)
    ]
    return train_df, val_df, test_df, rewritten_df


def get_classification_labels(df, args):
    if args.dataset != "political":
        labels = df[f"c{args.label_column}"].values
        labels = np.where(labels == "y", 1, 0)
    elif args.dataset == "political":
        labels = df[args.label_column].values
        labels = np.array([POLITICAL_LABEL_TO_INDEX[label] for label in labels])
    else:
        raise ValueError("Invalid dataset")
    return labels


def get_datasets(train_df, val_df, test_df, rewritten_df, args):
    datasets = DatasetDict(
        {
            df_name: Dataset.from_pandas(
                pd.DataFrame(
                    {
                        "text": df[args.text_column],
                        "label": get_classification_labels(df, args),
                    }
                )
            )
            for df_name, df in zip(
                ["train", "val", "test", "rewritten"],
                [train_df, val_df, test_df, rewritten_df],
            )
        }
    )
    return datasets


def tokenize_function(examples, tokenizer, args):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=dataset_to_max_length[args.dataset],
    )


def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    metric = load_metric("f1")
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    try:
        predictions = np.argmax(predictions, axis=1)

    except Exception as e:
        print(e)
        embed()
        exit()
    return {
        "f1": metric.compute(
            predictions=predictions, references=labels, average="macro"
        )["f1"]
    }


def main(args):
    log_file_name = create_log_file(args)
    train_df, val_df, test_df, rewritten_df = get_all_splits(args)

    datasets = get_datasets(train_df, val_df, test_df, rewritten_df, args)

    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    model = AutoModelForSequenceClassification.from_pretrained(
        "allenai/longformer-base-4096", num_labels=len(set(datasets["train"]["label"]))
    )

    tokenized_datasets = datasets.map(
        lambda examples: tokenize_function(examples, tokenizer, args),
        batched=True,
    )

    all_steps = len(tokenized_datasets["train"]) // args.batch_size * args.num_epochs
    logging_steps = 30
    print("Logging Steps: ", logging_steps)

    training_args = TrainingArguments(
        output_dir="./trained_models_longformer",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        logging_dir="./logs_models_longformer",
        report_to=None,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        logging_strategy="steps",
        save_steps=logging_steps,
        logging_steps=logging_steps,
        eval_steps=logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        learning_rate=1e-5,
        gradient_accumulation_steps=2,
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            transformers.EarlyStoppingCallback(
                early_stopping_patience=3, early_stopping_threshold=0.01
            )
        ],
    )

    print(trainer.evaluate(tokenized_datasets["test"]))
    trainer.train()
    print(trainer.evaluate(tokenized_datasets["test"]))

    # print the results of the best model
    logging.info("best score:")
    logging.info(trainer.evaluate(tokenized_datasets["val"])["eval_f1"])
    logging.info("best score on test essays")
    logging.info(trainer.evaluate(tokenized_datasets["test"])["eval_f1"])
    logging.info("complete report")
    extra_data = {}
    # put the test ids, with labels, and confidence scores
    extra_data["test"] = {
        "ids": test_df[args.id_column].values.tolist(),
        "labels": tokenized_datasets["test"]["label"],
        "predicted_labels": np.argmax(
            trainer.predict(tokenized_datasets["test"]).predictions, axis=1
        ).tolist(),
        "confidence_scores": trainer.predict(
            tokenized_datasets["test"]
        ).predictions.tolist(),
    }

    logging.info("-----REWRITTEN-----")
    logging.info("best score on rewritten essays")
    logging.info(trainer.evaluate(tokenized_datasets["rewritten"])["eval_f1"])
    extra_data["rewritten"] = {
        "ids": rewritten_df[args.id_column].values.tolist(),
        "labels": tokenized_datasets["rewritten"]["label"],
        "predicted_labels": np.argmax(
            trainer.predict(tokenized_datasets["rewritten"]).predictions, axis=1
        ).tolist(),
        "confidence_scores": trainer.predict(
            tokenized_datasets["rewritten"]
        ).predictions.tolist(),
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
        "--rewritten_input_file",
        type=str,
        required=False,
        help="Path to the CSV rewritten input file that has the same columns as the as the input file",
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
        default="longformer",
    )

    parser.add_argument("--features", type=str, default="longformer")

    parser.add_argument(
        "--output_type",
        type=str,
        default="classification",
    )

    parser.add_argument("--reverse", type=int, default=0)

    parser.add_argument(
        "--rewritten_type", type=str, required=False, default="syntax_grammar"
    )

    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()

    main(args)

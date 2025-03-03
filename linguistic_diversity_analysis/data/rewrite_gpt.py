import openai_handler
import pandas as pd
import time
import json
import os
from IPython import embed
from tqdm import tqdm
import argparse

prompts_for_questions = {
    "syntax_grammar": """Rewrite the following text using the best syntax and grammar and other revisions that are necessary: \"{}\"""",
    "rephrase": """Rephrase the following text: \"{}\"""",
}


def main(args):

    log_file_name = f"{args.dataset_name}_rewritten_{args.rewrite_type}_gpt.json"

    if os.path.exists(os.path.join(args.dataset_name, log_file_name)):
        with open(os.path.join(args.dataset_name, log_file_name), "r") as f:
            passages_rewritten = json.load(f)
    else:
        passages_rewritten = {}

    print("loaded passages_rewritten", len(passages_rewritten))

    if args.dataset_name == "reddit":
        df = pd.read_csv("reddit/filtered_comments.csv")
        # text columns is "body" and the id column is "name"
        df["id"] = df["name"]
        df["text"] = df["body"]
        df["id"] = df["id"].astype(str)
    elif args.dataset_name == "news":
        df = pd.read_csv("news/news_data.csv")
        # text column is "text", and since there's no id column, we'll use the index
        df["id"] = df.index
        df["id"] = df["id"].astype(str)
    elif args.dataset_name == "papers":
        df = pd.read_csv("papers/cl_cv_papers.csv")
        df["final_date"] = pd.to_datetime(df["update_date"])
        df["year"] = df["final_date"].dt.year
        df["month"] = df["final_date"].dt.month
        # text column is "abstract", and the id column is "id"
        df["text"] = df["abstract"]
        df["id"] = df["id"].astype(str)
    else:
        print("Invalid dataset_name")
        exit()

    # take a 200 sample randomly with a fixed seed for reproducibility,

    sampled_df = df.sample(n=200, random_state=42)
    second_sampled = df.sample(n=300, random_state=43)
    third_sampled = df.sample(n=300, random_state=44)
    df = pd.concat([sampled_df, second_sampled, third_sampled])

    curr_index = 0
    prog_bar = tqdm(total=len(df), leave=False)
    while curr_index != len(df):
        try:
            row = df.iloc[curr_index]
            row_id = row["id"]
            row_text = row["text"]
            if row_id in passages_rewritten:
                print("skipping", row_id)
                curr_index += 1
                prog_bar.update(1)
                continue
            if len(row_text.split()) > 2000:
                print("skipping (too long)", row_id)
                curr_index += 1
                prog_bar.update(1)
                continue

            prompt_filled = prompts_for_questions[args.rewrite_type].format(row_text)

            # print("prompt_filled: ", prompt_filled)
            # print("*******************")

            response = openai_handler.generate_chat_completion(
                system_content="You are an assistant that is obedient and helpful.",
                user_content=prompt_filled,
            )

            # print(response)
            # time.sleep(10)
            # print("*******************")

            # embed()

            passages_rewritten[str(row_id)] = {
                "original_text": row_text,
                "rewritten_text": response,
            }
            with open(os.path.join(args.dataset_name, log_file_name), "w") as f:
                json.dump(passages_rewritten, f)

            curr_index += 1
            prog_bar.update(1)
        except Exception as e:
            if "context_length_exceeded" in str(e):
                print("context_length_exceeded")
                curr_index += 1
                prog_bar.update(1)
                continue
            print(e)
            print("sleeping for 30 seconds")
            time.sleep(30)
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rewrite_type", type=str, required=True)
    parser.add_argument(
        "--dataset_name", type=str, required=True, choices=["reddit", "news", "papers"]
    )
    args = parser.parse_args()
    main(args)

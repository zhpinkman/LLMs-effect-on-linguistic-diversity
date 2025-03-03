import openai_handler
import pandas as pd
import time
import json
import os
from IPython import embed
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer
import transformers
import torch

prompts_for_questions = {
    "syntax_grammar": """Text: "{}"\nRewrite Text using the best syntax and grammar and other revisions that are necessary and return the result in \\boxed{{}}.""",
    "rephrase": """Text: "{}"\nRephrase Text and return the result in \\boxed{{}}.""",
}


# model = "meta-llama/Llama-2-13b-chat-hf"

# tokenizer = AutoTokenizer.from_pretrained(model)
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     torch_dtype=torch.float16,
#     device_map="auto",
# )


def main(args):
    log_file_name = f"{args.dataset_name}_rewritten_{args.rewrite_type}_llama.json"

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

            # embed()

            # print("prompt_filled: ", prompt_filled)
            # print("*******************")

            import requests

            url = "https://api.fireworks.ai/inference/v1/chat/completions"

            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt_filled,
                    }
                ],
                "model": "accounts/fireworks/models/llama-v3-70b-instruct",
                "stream": False,
                "n": 1,
                "max_tokens": 4096,
                "temperature": 0.0,
                "top_p": 1.0,
                "stop": [],
            }
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "authorization": "Bearer eCHxWHSMiS8DGPLV0vPg2ElEAEnbY5WGwFoL2UmAfE6GF3mC",
            }

            response = requests.post(url, json=payload, headers=headers)
            # print(response.text)
            if response.status_code == 400:
                if "too long" in response.json()["error"]["message"]:
                    print("skipping (too long)", row_id)
                    curr_index += 1
                    prog_bar.update(1)
                    continue

            # embed()
            # exit()
            response_text = response.json()["choices"][0]["message"]["content"]

            # response = pipeline(
            #     prompt_filled,
            #     do_sample=False,
            #     temperature=0.0,
            #     top_p=1.0,
            #     num_return_sequences=1,
            #     eos_token_id=tokenizer.eos_token_id,
            #     max_length=4096,
            # )[0]["generated_text"]

            # if "boxed" in response:
            #     print("boxed in response")
            #     # extract what is inside the boxed{} and return that
            #     updated_response = response.split("\\boxed{}")[1].split("\\boxed{")[1].split("}")[0]
            #     print("number of words in updated_response: ", len(updated_response.split()))
            # else:
            #     updated_response = ""

            # print("*******************")

            # print(response)
            # time.sleep(10)
            # print("*******************")

            # embed()
            # exit()

            passages_rewritten[str(row_id)] = {
                "original_text": row_text,
                "rewritten_text": response_text,
            }
            with open(
                os.path.join(
                    args.dataset_name,
                    f"{args.dataset_name}_rewritten_{args.rewrite_type}_llama.json",
                ),
                "w",
            ) as f:
                json.dump(passages_rewritten, f)

            curr_index += 1
            prog_bar.update(1)
        except Exception as e:
            print(e)
            curr_index += 1
            prog_bar.update(1)
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rewrite_type", type=str, required=True)
    parser.add_argument(
        "--dataset_name", type=str, required=True, choices=["reddit", "news", "papers"]
    )
    args = parser.parse_args()
    main(args)

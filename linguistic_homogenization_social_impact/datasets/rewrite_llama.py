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
    if os.path.exists(
        os.path.join(
            args.output_dir,
            f"{args.dataset_name}_rewritten_{args.rewrite_type}_llama.json",
        )
    ):
        with open(
            os.path.join(
                args.output_dir,
                f"{args.dataset_name}_rewritten_{args.rewrite_type}_llama.json",
            ),
            "r",
        ) as f:
            passages_rewritten = json.load(f)
    else:
        passages_rewritten = {}

    print("loaded passages_rewritten", len(passages_rewritten))

    df = pd.read_csv(args.input_file, encoding="mac_roman")
    df[args.id_column] = df[args.id_column].astype(str)

    curr_index = 0
    prog_bar = tqdm(total=len(df), leave=False)
    while curr_index != len(df):
        try:
            row = df.iloc[curr_index]
            row_id = row[args.id_column]
            row_text = row[args.text_column]
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
                "model": "accounts/fireworks/models/llama-v2-70b-chat",
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

            passages_rewritten[str(row_id)] = response_text
            with open(
                os.path.join(
                    args.output_dir,
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
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--rewrite_type", type=str, required=True)
    parser.add_argument("--id_column", type=str, required=True)
    parser.add_argument("--text_column", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    args = parser.parse_args()
    main(args)

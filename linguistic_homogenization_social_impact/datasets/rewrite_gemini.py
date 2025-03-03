import openai_handler
import pandas as pd
import time
import json
import os
from IPython import embed
from tqdm import tqdm
import argparse
import vertexai
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig

prompts_for_questions = {
    "syntax_grammar": """Text: "{}"\nRewrite Text using the best syntax and grammar and other revisions that are necessary and return the result in \\boxed{{}}.\n""",
    "rephrase": """Text: "{}"\nRephrase Text and return the result in \\boxed{{}}.\n""",
}


vertexai.init(project="gemininonverbal", location="us-central1")
GEMINI_MODEL = GenerativeModel("gemini-pro")


def organize_prompt(prompt):
    return


def prompt_model(prompt_filled):
    x = GEMINI_MODEL.generate_content(
        prompt_filled,
        generation_config=GenerationConfig(
            temperature=0.0,
            top_p=1.0,
            candidate_count=1,
            max_output_tokens=4096,
        ),
    )
    return x


def main(args):
    log_file_name = f"{args.dataset_name}_rewritten_{args.rewrite_type}_gemini.json"
    if os.path.exists(
        os.path.join(
            args.output_dir,
            log_file_name,
        )
    ):
        with open(
            os.path.join(
                args.output_dir,
                log_file_name,
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

            response = prompt_model(prompt_filled)

            passages_rewritten[str(row_id)] = response.text
            with open(
                os.path.join(
                    args.output_dir,
                    log_file_name,
                ),
                "w",
            ) as f:
                json.dump(passages_rewritten, f)

            curr_index += 1
            prog_bar.update(1)
        except Exception as e:
            print(e)
            print(response)
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

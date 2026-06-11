import argparse
import pandas as pd
import os
from tqdm import tqdm
import requests
import json
from IPython import embed


def get_embeddings(list_of_ids, list_of_texts, results_embeddings={}):
    for _id, essay in tqdm(zip(list_of_ids, list_of_texts), total=len(list_of_ids)):
        try:
            if str(_id) in results_embeddings.keys():
                continue
            api_key = os.environ.get("OPENAI_API_KEY")
            url = "https://api.openai.com/v1/embeddings"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            data = {
                "input": essay,
                "model": "text-embedding-ada-002",
                "encoding_format": "float",
            }

            # Send the POST request
            response = requests.post(url, headers=headers, json=data)

            # Check the response status and content
            if response.status_code == 200:
                # Successful request, you can access the response data using response.json()
                response_data = response.json()
                results_embeddings[str(_id)] = response_data
            else:
                # Request failed, print the error message
                print(
                    f"Request failed with status code {response.status_code}: {response.text}"
                )
                continue
        except Exception as e:
            print(e)
            continue
    return results_embeddings


def main(args):
    output_file = (args.input_file.split("/")[-1]).split(".")[0] + "_embeddings.json"
    if os.path.exists(os.path.join(args.output_dir, output_file)):
        with open(os.path.join(args.output_dir, output_file)) as f:
            results_embeddings = json.load(f)
            print("loaded embeddings", len(results_embeddings))
    else:
        results_embeddings = {}

    df = pd.read_csv(args.input_file)
    list_of_ids = df[args.id_column].values
    list_of_texts = df[args.text_column].values
    results_embeddings = get_embeddings(list_of_ids, list_of_texts, results_embeddings)
    with open(os.path.join(args.output_dir, output_file), "w") as f:
        json.dump(results_embeddings, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--id_column", type=str, required=True)
    parser.add_argument("--text_column", type=str, required=True)

    args = parser.parse_args()
    main(args)

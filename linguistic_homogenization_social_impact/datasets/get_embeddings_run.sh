dataset=$1
original_file=$2
id_column=$3

python get_embeddings.py \
    --input_file ${dataset}/${original_file}.csv \
    --output_dir ${dataset} \
    --id_column ${id_column} \
    --text_column text

for LLM_MODEL in "gemini" "gpt" "llama"; do
    python get_embeddings.py \
        --input_file ${dataset}/${dataset}_rewritten_syntax_grammar_${LLM_MODEL}_cleaned.csv \
        --output_dir ${dataset} \
        --id_column ${id_column} \
        --text_column text

    python get_embeddings.py \
        --input_file ${dataset}/${dataset}_rewritten_rephrase_${LLM_MODEL}_cleaned.csv \
        --output_dir ${dataset} \
        --id_column ${id_column} \
        --text_column text
done

# python get_embeddings.py \
#     --input_file essays/essays_rewritten_syntax_grammar_${LLM_MODEL}_cleaned.csv \
#     --output_dir essays \
#     --id_column "#AUTHID" \
#     --text_column text

# python get_embeddings.py \
#     --input_file essays/essays_rewritten_rephrase_${LLM_MODEL}_cleaned.csv \
#     --output_dir essays \
#     --id_column "#AUTHID" \
#     --text_column text

# python get_embeddings.py \
#     --input_file wassa/wassa_rewritten_syntax_grammar_${LLM_MODEL}_cleaned.csv \
#     --output_dir wassa \
#     --id_column id \
#     --text_column text

# python get_embeddings.py \
#     --input_file wassa/wassa_rewritten_rephrase_${LLM_MODEL}_cleaned.csv \
#     --output_dir wassa \
#     --id_column id \
#     --text_column text

prompt=$1

# if prompt == syntax_grammar or rephrase
if [ $prompt == "syntax_grammar" ]; then
    python rewrite_llama.py \
        --input_file political/clean_data.csv \
        --output_dir political/ \
        --rewrite_type syntax_grammar \
        --id_column id \
        --text_column text \
        --dataset_name political
elif [ $prompt == "rephrase" ]; then
    python rewrite_llama.py \
        --input_file political/clean_data.csv \
        --output_dir political/ \
        --rewrite_type rephrase \
        --id_column id \
        --text_column text \
        --dataset_name political
fi

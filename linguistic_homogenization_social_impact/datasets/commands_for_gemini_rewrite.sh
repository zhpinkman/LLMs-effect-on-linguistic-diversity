python rewrite_gemini.py \
    --input_file political/clean_data.csv \
    --output_dir political/ \
    --rewrite_type syntax_grammar \
    --id_column id \
    --text_column text \
    --dataset_name political

python rewrite_gemini.py \
    --input_file political/clean_data.csv \
    --output_dir political/ \
    --rewrite_type rephrase \
    --id_column id \
    --text_column text \
    --dataset_name political

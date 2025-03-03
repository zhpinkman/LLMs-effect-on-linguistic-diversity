text_column="text"
dataset=$1
gpus=$2
batch_size=$3

echo "Running for $1 dataset"
echo "Using GPU: $2"
echo "Batch size: $3"

if [ "$1" == "essays" ]; then
    echo "ESSAYS"
    for num in {1..10}; do
        for LLM_model in "gpt" "llama" "gemini"; do
            for rewritten_type in "syntax_grammar_${LLM_model}" "rephrase_${LLM_model}"; do
                input_file="../datasets/essays/essays_anon_full.csv"
                rewritten_input_file="../datasets/essays/essays_rewritten_${rewritten_type}_cleaned.csv"
                id_column="#AUTHID"
                for label_column in "EXT" "NEU" "AGR" "CON" "OPN"; do
                    for reverse in 0 1; do
                        CUDA_VISIBLE_DEVICES=${gpus} python longformer_model.py \
                            --input_file $input_file \
                            --rewritten_input_file $rewritten_input_file \
                            --text_column $text_column \
                            --id_column $id_column \
                            --dataset $dataset \
                            --label_column $label_column \
                            --rewritten_type ${rewritten_type} \
                            --LLM_model $LLM_model \
                            --batch_size $batch_size \
                            --reverse $reverse
                    done
                done
            done
        done
    done
elif [ "$1" == "wassa" ]; then
    echo "WASSA"
    for num in {1..10}; do
        for LLM_model in "gpt" "llama" "gemini"; do
            for rewritten_type in "syntax_grammar_${LLM_model}" "rephrase_${LLM_model}"; do
                input_file="../datasets/wassa/clean_wassa.csv"
                rewritten_input_file="../datasets/wassa/wassa_rewritten_${rewritten_type}_cleaned.csv"
                id_column="id"
                for label_column in ".iri.perspective" ".iri.distress" ".iri.fantasy" ".iri.concern"; do
                    for reverse in 0 1; do
                        CUDA_VISIBLE_DEVICES=${gpus} python longformer_model.py \
                            --input_file $input_file \
                            --rewritten_input_file $rewritten_input_file \
                            --text_column $text_column \
                            --id_column $id_column \
                            --dataset $dataset \
                            --label_column $label_column \
                            --rewritten_type ${rewritten_type} \
                            --LLM_model $LLM_model \
                            --batch_size $batch_size \
                            --reverse $reverse
                    done
                done
            done
        done
    done
elif [ "$1" == "facebook" ]; then
    echo "FACEBOOK"
    for num in {1..10}; do
        for LLM_model in "gpt" "llama" "gemini"; do
            for rewritten_type in "syntax_grammar_${LLM_model}" "rephrase_${LLM_model}"; do
                input_file="../datasets/facebook/full_dataset_clean.csv"
                rewritten_input_file="../datasets/facebook/facebook_rewritten_${rewritten_type}_cleaned.csv"
                id_column="subject_id"
                for label_column in ".fairness" ".care" ".loyalty" ".authority" ".purity"; do
                    for reverse in 0 1; do
                        CUDA_VISIBLE_DEVICES=${gpus} python longformer_model.py \
                            --input_file $input_file \
                            --rewritten_input_file $rewritten_input_file \
                            --text_column $text_column \
                            --id_column $id_column \
                            --dataset $dataset \
                            --label_column $label_column \
                            --rewritten_type ${rewritten_type} \
                            --LLM_model $LLM_model \
                            --batch_size $batch_size \
                            --reverse $reverse
                    done
                done
            done
        done
    done
elif [ "$1" == "political" ]; then
    echo "POLITICAL"
    for num in {1..20}; do
        for LLM_model in "gpt" "llama" "gemini"; do
            for rewritten_type in "syntax_grammar_${LLM_model}" "rephrase_${LLM_model}"; do
                input_file="../datasets/political/clean_data_agg.csv"
                rewritten_input_file="../datasets/political/political_rewritten_${rewritten_type}_cleaned.csv"
                id_column="speakerid"
                for reverse in 0 1; do
                    for label_column in "cohort" "party" "gender"; do
                        CUDA_VISIBLE_DEVICES=${gpus} python longformer_model.py \
                            --input_file $input_file \
                            --rewritten_input_file $rewritten_input_file \
                            --text_column $text_column \
                            --id_column $id_column \
                            --dataset $dataset \
                            --label_column $label_column \
                            --rewritten_type ${rewritten_type} \
                            --LLM_model $LLM_model \
                            --batch_size $batch_size \
                            --reverse $reverse
                    done
                done
            done
        done
    done
else
    echo "Invalid dataset"
fi

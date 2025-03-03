text_column="text"
echo "Mode" $1
# # # # # # # # # # # # # # # # # #  ESSAYS  # # # # # # # # # # # # # # # # # #
if [ "$1" == "essays" ]; then
    echo "ESSAYS"
    for num in {1..30}; do
        for LLM_model in "gpt" "llama" "gemini"; do
            for rewritten_type in "syntax_grammar_${LLM_model}" "rephrase_${LLM_model}"; do
                input_file="../datasets/essays/essays_anon_full.csv"
                input_file_embeddings="../datasets/essays/essays_anon_full_embeddings.json"
                rewritten_input_file="../datasets/essays/essays_rewritten_${rewritten_type}_cleaned.csv"
                rewritten_input_file_embeddings="../datasets/essays/essays_rewritten_${rewritten_type}_cleaned_embeddings.json"
                id_column="#AUTHID"
                dataset="essays"
                for label_column in "EXT" "NEU" "AGR" "CON" "OPN"; do
                    for features in "openai" "tfidf"; do
                        for model_type in "gradient_boosting" "svm" "random_forest" "regression"; do
                            for output_type in "classification"; do
                                python model_with_aug.py \
                                    --input_file $input_file \
                                    --input_file_embeddings $input_file_embeddings \
                                    --rewritten_input_file $rewritten_input_file \
                                    --rewritten_input_file_embeddings $rewritten_input_file_embeddings \
                                    --text_column $text_column \
                                    --id_column $id_column \
                                    --dataset $dataset \
                                    --features $features \
                                    --label_column $label_column \
                                    --model_type $model_type \
                                    --output_type $output_type \
                                    --rewritten_type ${rewritten_type} \
                                    --LLM_model $LLM_model
                            done
                        done
                    done
                done
            done
        done
    done
elif [ "$1" == "wassa" ]; then
    echo "WASSA"
    for num in {1..50}; do
        for LLM_model in "gpt" "llama" "gemini"; do
            for rewritten_type in "syntax_grammar_${LLM_model}" "rephrase_${LLM_model}"; do
                input_file="../datasets/wassa/clean_wassa.csv"
                input_file_embeddings="../datasets/wassa/clean_wassa_embeddings.json"
                rewritten_input_file="../datasets/wassa/wassa_rewritten_${rewritten_type}_cleaned.csv"
                rewritten_input_file_embeddings="../datasets/wassa/wassa_rewritten_${rewritten_type}_cleaned_embeddings.json"
                id_column="id"
                dataset="wassa"
                for label_column in ".iri.perspective" ".iri.distress" ".iri.fantasy" ".iri.concern"; do
                    for features in "tfidf" "openai"; do
                        for model_type in "gradient_boosting" "svm" "random_forest" "regression"; do
                            for output_type in "classification"; do
                                python model_with_aug.py \
                                    --input_file $input_file \
                                    --input_file_embeddings $input_file_embeddings \
                                    --rewritten_input_file $rewritten_input_file \
                                    --rewritten_input_file_embeddings $rewritten_input_file_embeddings \
                                    --text_column $text_column \
                                    --id_column $id_column \
                                    --dataset $dataset \
                                    --features $features \
                                    --label_column $label_column \
                                    --model_type $model_type \
                                    --output_type $output_type \
                                    --rewritten_type ${rewritten_type} \
                                    --LLM_model $LLM_model
                            done
                        done
                    done
                done
            done
        done
    done
elif [ "$1" == "facebook" ]; then
    echo "FACEBOOK"
    for num in {1..30}; do
        for LLM_model in "gpt" "llama" "gemini"; do
            for rewritten_type in "syntax_grammar_${LLM_model}" "rephrase_${LLM_model}"; do
                input_file="../datasets/facebook/full_dataset_clean.csv"
                input_file_embeddings="../datasets/facebook/full_dataset_clean_embeddings.json"
                rewritten_input_file="../datasets/facebook/facebook_rewritten_${rewritten_type}_cleaned.csv"
                rewritten_input_file_embeddings="../datasets/facebook/facebook_rewritten_${rewritten_type}_cleaned_embeddings.json"
                id_column="subject_id"
                dataset="facebook"
                for label_column in ".fairness" ".care" ".loyalty" ".authority" ".purity"; do
                    for features in "tfidf" "openai"; do
                        for model_type in "gradient_boosting" "svm" "random_forest" "regression"; do
                            for output_type in "classification"; do
                                python model_with_aug.py \
                                    --input_file $input_file \
                                    --input_file_embeddings $input_file_embeddings \
                                    --rewritten_input_file $rewritten_input_file \
                                    --rewritten_input_file_embeddings $rewritten_input_file_embeddings \
                                    --text_column $text_column \
                                    --id_column $id_column \
                                    --dataset $dataset \
                                    --features $features \
                                    --label_column $label_column \
                                    --model_type $model_type \
                                    --output_type $output_type \
                                    --rewritten_type ${rewritten_type} \
                                    --LLM_model $LLM_model
                            done
                        done
                    done
                done
            done
        done
    done
elif [ "$1" == "political" ]; then
    echo "POLITICAL"
    for num in {1..30}; do
        for LLM_model in "gpt" "llama" "gemini"; do
            for rewritten_type in "syntax_grammar_${LLM_model}" "rephrase_${LLM_model}"; do
                input_file="../datasets/political/clean_data_agg.csv"
                input_file_embeddings="../datasets/political/clean_data_agg_embeddings.json"
                rewritten_input_file="../datasets/political/political_rewritten_${rewritten_type}_cleaned.csv"
                rewritten_input_file_embeddings="../datasets/political/political_rewritten_${rewritten_type}_cleaned_embeddings.json"
                id_column="speakerid"
                dataset="political"
                for features in "tfidf" "openai"; do
                    for model_type in "gradient_boosting" "svm" "random_forest" "regression"; do
                        for output_type in "classification"; do
                            for reverse in 0 1; do
                                for label_column in "cohort" "party" "gender"; do
                                    python model_with_aug.py \
                                        --input_file $input_file \
                                        --input_file_embeddings $input_file_embeddings \
                                        --rewritten_input_file $rewritten_input_file \
                                        --rewritten_input_file_embeddings $rewritten_input_file_embeddings \
                                        --text_column $text_column \
                                        --id_column $id_column \
                                        --dataset $dataset \
                                        --features $features \
                                        --label_column $label_column \
                                        --model_type $model_type \
                                        --output_type $output_type \
                                        --rewritten_type ${rewritten_type} \
                                        --LLM_model $LLM_model
                                done
                            done
                        done
                    done
                done
            done
        done
    done
fi

# inference ODQA for evaluation with dev dataset

python inference.py \
--output_dir="./eval" \
--model_name_or_path="./finetune/" \
--dataset_name="../data/train_data" \
--do_eval
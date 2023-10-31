export CUDA_VISIBLE_DEVICES=4

#### Get script path and name
SCRIPT_PATH=$(readlink -f "$0")
FILE_NAME="$(basename $SCRIPT_PATH)"

dataset="pens_rec_own"
max_source_length=24

#// Train recommendation model
#// When training:
#model_name_or_path=facebook/bart-base
#output_dir=../results/pens_rec_own/bart_rec_ctr_eps1e-7_normalize

#// When testing:
#model_name_or_path=../results/pens_rec_own/bart_rec_ctr_eps1e-7/checkpoint-95000
#model_name_or_path=../results/pens_rec_own/bart_rec/checkpoint-179000
#model_name_or_path=../results/pens_rec_own/bart_rec/checkpoint-110000
model_name_or_path=../results/pens_rec_own/bart_rec_ctr_eps1e-7_normalize/checkpoint-110000
output_dir=$model_name_or_path

mkdir -p $output_dir
cp $SCRIPT_PATH $output_dir/$FILE_NAME

python main.py\
  --do_train false\
  --do_eval false\
  --do_predict true\
  --dataset_name $dataset\
  --max_source_length $max_source_length\
  --pad_to_max_length true\
  --overwrite_output_dir true\
  --overwrite_cache true\
  --remove_unused_columns false\
  --model_name_or_path $model_name_or_path\
  --output_dir $output_dir\
  --lr_scheduler_type constant\
  --learning_rate 3e-6\
  --gradient_accumulation_steps 1\
  --per_device_train_batch_size 32\
  --per_device_eval_batch_size 64\
  --max_eval_samples 2000\
  --num_train_epochs 15\
  --eval_steps 1000\
  --save_steps 1000\
  --evaluation_strategy steps\
  --save_strategy steps\
  --save_total_limit 1\
  --report_to tensorboard\
  --logging_steps 40\
  --ctr_loss true\
  --predict_with_generate true\
  --save_model_accord_to_rouge true\
  --label_names ["labels_rec"]\


#echo "===== Start Running ====="
#model_name_or_path="../datasets/user_feat/REC/checkpoint-47500"
##model_name_or_path="facebook/bart-base"
#
#output_dir="../datasets/user_feat/REC/checkpoint-47500"
#
#echo "Get news features from $output_dir"
#echo "output_dir $output_dir"
#
#mkdir -p $output_dir
#cp $SCRIPT_PATH $output_dir/$FILE_NAME
#
##// Get get news features for the news headlines
#python main.py\
#  --do_predict true\
#  --get_news_features true\
#  --test_file "../datasets/pens/news.pkl"\
#  --dataset_name $dataset\
#  --max_source_length $max_source_length\
#  --pad_to_max_length true\
#  --per_device_eval_batch_size 32\
#  --overwrite_output_dir true\
#  --overwrite_cache true\
#  --remove_unused_columns false\
#  --model_name_or_path $model_name_or_path\
#  --output_dir $output_dir/editor_headline_v2\
#
#// Get get news features for the 1-st prediction outputs
#python main.py\
#  --do_predict true\
#  --get_news_features true\
#  --test_file "../results/pens_ghg_own/bart/checkpoint-98000/all_news/generated_predictions.txt"\
#  --dataset_name $dataset\
#  --max_source_length $max_source_length\
#  --pad_to_max_length true\
#  --per_device_eval_batch_size 32\
#  --overwrite_output_dir true\
#  --overwrite_cache true\
#  --remove_unused_columns false\
#  --model_name_or_path $model_name_or_path\
#  --output_dir $output_dir/first_stage_v2\
#

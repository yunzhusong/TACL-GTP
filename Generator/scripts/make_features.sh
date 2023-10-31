export CUDA_VISIBLE_DEVICES=1

#### Get script path and name
SCRIPT_PATH=$(readlink -f "$0")
FILE_NAME="$(basename $SCRIPT_PATH)"

dataset="pens_rec_own"
#model_name_or_path="../datasets/user_feat/REC/checkpoint-47500"
#model_name_or_path="facebook/bart-base"

output_dir="../datasets/user_feat/REC/checkpoint-47500"

echo "Get news features from $output_dir"
echo "output_dir $output_dir"
echo "===== Start Running ====="

mkdir -p $output_dir
cp $SCRIPT_PATH $output_dir/$FILE_NAME

max_source_length=24

# For evaluation
python main.py\
  --do_predict true\
  --get_news_features true\
  --test_file "../datasets/pens/news.pkl"\
  --dataset_name $dataset\
  --max_source_length $max_source_length\
  --pad_to_max_length true\
  --per_device_eval_batch_size 32\
  --overwrite_output_dir true\
  --overwrite_cache true\
  --remove_unused_columns false\
  --model_name_or_path $model_name_or_path\
  --output_dir $output_dir\

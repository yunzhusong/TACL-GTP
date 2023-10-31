export CUDA_VISIBLE_DEVICES=5

#### Get script path and name
SCRIPT_PATH=$(readlink -f "$0")
FILE_NAME="$(basename $SCRIPT_PATH)"

dataset="pens_rec_own"
model_name_or_path="../datasets/user_feat/REC/checkpoint-47500"
#model_name_or_path="facebook/bart-base"

echo "===== Start Running ====="

first_stage_file="../results/pens_ghg_own/bart/checkpoint-98000/phg"
cp $SCRIPT_PATH $first_stage_file/$FILE_NAME

# For evaluation
python3 main.py\
  --do_predict true\
  --get_recommended_score true\
  --news_file "../datasets/pens/news.pkl"\
  --test_file "../datasets/specialize_own/test.pkl"\
  --pred_file $first_stage_file\
  --pred_file_is_full true\
  --take_pred_text true\
  --user_name NT$userwise_index\
  --dataset_name $dataset\
  --max_source_length 24\
  --pad_to_max_length true\
  --per_device_eval_batch_size 32\
  --overwrite_output_dir true\
  --overwrite_cache true\
  --remove_unused_columns false\
  --model_name_or_path $model_name_or_path\
  --output_dir $first_stage_file\

#model_dir="../results/s2_own/bart_input_pred_A_extra_online_A_userize_average_mum_loss/checkpoint-9000/userwise_own/userize_seed0_t80_v20_t100/ape"
#prediction_dir="$model_dir/ape"
#
#cp $SCRIPT_PATH $prediction_dir/$FILE_NAME
#
#
## 1. Format the prediction results
#pathon ape/file_format.py\
#  --model_path $model_dir\
#  --out_path $prediction_dir\
#  --num_user 60\
#
# 2. Run recommendation model
#for ((userwise_index=1; userwise_index<=60; userwise_index++));
#do
#	# For evaluation
#	python3 main.py\
#	  --do_predict true\
#	  --get_recommended_score true\
#	  --news_file "../datasets/pens/news.pkl"\
#	  --test_file "../datasets/specialize_own/test.pkl"\
#	  --pred_file $prediction_dir/NT$userwise_index\
#	  --pred_file_is_full false\
#	  --take_pred_text true\
#	  --user_name NT$userwise_index\
#	  --dataset_name $dataset\
#	  --max_source_length 24\
#	  --pad_to_max_length true\
#	  --per_device_eval_batch_size 32\
#	  --overwrite_output_dir true\
#	  --overwrite_cache true\
#	  --remove_unused_columns false\
#	  --model_name_or_path $model_name_or_path\
#	  --output_dir $prediction_dir/NT$userwise_index\
#
#done

# 3. Add the model_path to the following file and run it
#python anal/interpret_results.py

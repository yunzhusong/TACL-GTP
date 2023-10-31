export CUDA_VISIBLE_DEVICES=0

#### Get script path and name
SCRIPT_PATH=$(readlink -f "$0")
FILE_NAME="$(basename $SCRIPT_PATH)"

dataset="s2_own"
model_dir="../results/s2_own"
out_dir="../results/s2_own_userize"
log_dir="../results/s2_own_userize/logging"

userize_ufeat_path="../datasets/user_feat/REC/checkpoint-47500"
#userize_ufeat_path="../datasets/user_feat/bart"
predict_txt_file="../results/pens_ghg_own/bart/checkpoint-98000/all_news/generated_predictions.txt"

model_name_or_path="bart_input_pred_A_extra_online_A_userize/checkpoint-8200"

max_source_length=256 # 256 or 224
max_target_length=64
batch_size=16
accumulate=1

extra_info=false
retrieve_online=true

userwise=true
userwise_seed=0
userwise_sample_type=informativeness
seed="seed$userwise_seed"
#split="t10_v90_t100"
#split="t20_v80_t100"
split="t80_v20_t100"

# ===== Start ==== # 
userize=true
#userize_ufeat_type="spectral_cluster_centroid_news"
userize_ufeat_type="average_news"
#userize_ufeat_type="closest_news"
#userize_ufeat_type="user_feat"
#userize_ufeat_type="tail_feat"
userize_complex_proj=false
userize_type_embedding=false
userize_loss=false
userize_pum=false
userize_ctr_threshold=37 #37 #215 #275 #930

#other="_dev"
#other="_userfeat_NoUserloss_dev"
other="_average_NoUserloss_dev"
#other="_tailfeat_$userize_ctr_threshold-NoUserloss_dev"
#other="_tailfeat_$userize_ctr_threshold-NoUserloss_bart_news_dev"

if [[ $userize == true && $extra_info == true ]]; then
  prefix="$split/userize_$userwise_sample_type$other"
elif [[ $userize == true ]]; then
  prefix="$split/userize_no_extra_$userwise_sample_type$other"
elif [[ $extra_info == true ]]; then
  prefix="$split/$userwise_sample_type$other"
else
  prefix="$split/no_extra_$userwise_sample_type$other"
fi

mkdir -p $out_dir/$model_name_or_path/$prefix
cp $SCRIPT_PATH $out_dir/$model_name_or_path/$prefix/$FILE_NAME

for ((userwise_index=81; userwise_index<=93; userwise_index++));
do
	exp_name="$prefix/index$userwise_index"
	mkdir -p $out_dir/$model_name_or_path/$exp_name

	echo "Running $FILE_NAME"
	echo "===== Start Finetuning ====="
	echo "loading         | $model_name_or_path"
	echo "experiment Name | $exp_name"
	echo "extra_info      | $extra_info"
	echo "userize         | $userize"
	echo "userwise        | $userwise"

	# For finetuning
	python main.py\
	  --do_finetune true\
	  --do_train true\
	  --do_eval true\
	  --do_predict true\
	  --find_best_checkpoint true\
	  --remove_after_predict true\
	  --dataset_name $dataset\
	  --max_source_length $max_source_length\
	  --max_target_length $max_target_length\
	  --generation_max_length $max_target_length\
	  --pad_to_max_length true\
	  --per_device_train_batch_size $batch_size\
	  --per_device_eval_batch_size $batch_size\
	  --gradient_accumulation_steps $accumulate\
	  --num_train_epochs 100\
	  --eval_steps 20\
	  --save_steps 20\
	  --evaluation_strategy steps\
	  --lr_scheduler_type linear\
	  --learning_rate 3e-6\
	  --warmup_steps 150\
	  --save_strategy steps\
	  --save_total_limit 1\
	  --save_model_accord_to_rouge true\
	  --predict_with_generate true\
	  --report_to tensorboard\
	  --logging_steps 5\
	  --overwrite_output_dir true\
	  --overwrite_cache true\
	  --remove_unused_columns false\
	  --model_name_or_path $model_dir/$model_name_or_path\
	  --output_dir $out_dir/$model_name_or_path/$exp_name\
	  --logging_dir $log_dir/$model_name_or_path/$exp_name/log\
	  --text_file $predict_txt_file\
	  --corrupt false\
	  --corrupt_token_infilling false\
	  --corrupt_max_mask_num 3\
	  --corrupt_max_mask_length_lambda 3\
	  --corrupt_token_shuffle false\
	  --userize $userize\
	  --userize_ufeat_path $userize_ufeat_path\
	  --userize_user_token_length 5\
	  --userize_loss $userize_loss\
	  --userize_mum false\
	  --userize_pum $userize_pum\
	  --userize_ufeat_type $userize_ufeat_type\
	  --userize_complex_proj $userize_complex_proj\
	  --userize_type_embedding $userize_type_embedding\
	  --extra_info $extra_info\
	  --retrieve_online $retrieve_online\
	  --userwise $userwise\
	  --userwise_index $userwise_index\
	  --userwise_split "80 20 100"\
	  --userwise_seed $userwise_seed\
	  --userwise_sample_type $userwise_sample_type\
          --userize_ctr_threshold=$userize_ctr_threshold\
	  --developing true\

done
# ===== END ==== #

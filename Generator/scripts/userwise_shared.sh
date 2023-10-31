export CUDA_VISIBLE_DEVICES=3

#### Get script path and name
SCRIPT_PATH=$(readlink -f "$0")
FILE_NAME="$(basename $SCRIPT_PATH)"

dataset="s2_shared_own"
model_dir="../results/s2_shared_own"
out_dir="../results/s2_shared_own_userize"
log_dir="../results/s2_shared_own_userize/logging"

userize_ufeat_path="../datasets/user_feat/REC/checkpoint-47500"
predict_txt_file="../results/pens_ghg_own/bart/checkpoint-98000/all_news/generated_predictions.txt"

#model_name_or_path="bart-base"
#model_name_or_path="bart_input_pred/checkpoint-9400"
#model_name_or_path="bart_input_pred_A_userize_mum_loss/checkpoint-11400"
#model_name_or_path="bart_input_pred_A_extra_online/checkpoint-9000"
#model_name_or_path="bart_input_pred_A_extra_online_A_userize_mum_loss/checkpoint-9000"
#model_name_or_path="bart_input_pred_A_extra_online_A_userize_average_mum_loss/checkpoint-9000"
#model_name_or_path="bart_input_pred_A_extra_online_A_userize_closest_mum_loss/checkpoint-9000"
#model_name_or_path="bart_input_pred_A_extra_online_A_userize_mum_loss_10token/checkpoint-8200"
#model_name_or_path="bart_input_pred_A_extra_online_A_cpxtype_userize_average_mum_loss/checkpoint-8200"
#model_name_or_path="bart_input_pred_A_extra_online_A_cpxtype_userize_kmeans_mum_loss/checkpoint-8200"
#model_name_or_path="bart_input_pred_A_extra_online_A_cpxtype_userize_spectral_mum_loss/checkpoint-8200"
#model_name_or_path="bart_input_pred_A_extra_online_A_cpx_userize_average_mum_loss/checkpoint-8200"
#model_name_or_path="bart_input_pred_A_extra_online_A_cpx_userize_closest_mum_loss_v2/checkpoint-8200"
#model_name_or_path="bart_input_pred_A_extra_online_A_cpx_userize_spectral_mum_loss_v2/checkpoint-8200"
#model_name_or_path="bart_input_pred_A_extra_online_A_userize_spectral_centroid_mum_loss_v3/checkpoint-6800"
#model_name_or_path="bart_input_pred_A_extra_online_A_userize_average_mum_loss_v3/checkpoint-9000"
#model_name_or_path="bart_input_pred_A_userize_spectral_centroid_mum_loss_v3/checkpoint-9400"
#model_name_or_path="bart_input_pred_A_extra_online_A_type_userize_average_mum_sum_loss_v3/checkpoint-8200"
#model_name_or_path="bart_input_pred_A_extra_online_A_type_userize_average_pum_loss/checkpoint-13000"
#model_name_or_path="bart_input_pred_A_extra_online_A_userize_average_ptum_loss/checkpoint-12500"
#model_name_or_path="bart_input_pred_A_extra_online_A_userize_closest_ptum_loss/checkpoint-12500"
#model_name_or_path="init_ghg/bart_input_pred_A_extra_online_A_userize_closest_ptum_loss/checkpoint-12500"
#model_name_or_path="init_ghg/bart_input_pred_A_extra_online_A_userize_userfeat_ptum_loss/checkpoint-12500"
#model_name_or_path="bart_input_pred_A_extra_online_A_userize_tailfeat_ptum_loss/checkpoint-12500"
model_name_or_path="/bart_input_pred_A_extra_online_A_userize_average_mum_loss_v4/checkpoint-8500"

max_source_length=256 # 256 or 224
max_target_length=64
batch_size=16
accumulate=4

extra_info=true
retrieve_online=true

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
userize_ctr_threshold=20 #37 #215 #275 #930

userwise=true
userwise_seed=0
userwise_split="80 20 100"

split="_t80_v20_t100"
seed="seed$userwise_seed"
other="_average_NoUserloss"
#other="_tailfeat_$userize_ctr_threshold-NoUserloss"

if [[ $userize == true && $extra_info == true ]]; then
  prefix="userize_$seed$split$other"
elif [[ $userize == true ]]; then
  prefix="userize_no_extra_$seed$split$other"
elif [[ $extra_info == true ]]; then
  prefix="$seed$split$other"
else
  prefix="no_extra_$seed$split$other"
fi

mkdir -p $out_dir/$model_name_or_path/$prefix
cp $SCRIPT_PATH $out_dir/$model_name_or_path/$prefix/$FILE_NAME

for ((userwise_index=20; userwise_index<=102; userwise_index++));
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
	echo "userwise_split  | $userwise_split"

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
	  --num_train_epochs 200\
	  --eval_steps 10\
	  --save_steps 10\
	  --evaluation_strategy steps\
	  --lr_scheduler_type constant\
	  --learning_rate 1e-6\
	  --warmup_steps 100\
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
          --userize_ctr_threshold=$userize_ctr_threshold\

done
# ===== END ==== #
# ===== Start ==== # 
userize=true
#userize_ufeat_type="spectral_cluster_centroid_news"
#userize_ufeat_type="average_news"
#userize_ufeat_type="closest_news"
#userize_ufeat_type="user_feat"
userize_ufeat_type="tail_feat"
userize_complex_proj=false
userize_type_embedding=false
userize_loss=false
userize_pum=false
userize_ctr_threshold=20 #37 #215 #275 #930

userwise=true
userwise_seed=0
userwise_split="80 20 100"

split="_t80_v20_t100"
seed="seed$userwise_seed"
#other="_tailfeat_$userize_ctr_threshold-NoUserloss"
other=""

if [[ $userize == true && $extra_info == true ]]; then
  prefix="userize_$seed$split$other"
elif [[ $userize == true ]]; then
  prefix="userize_no_extra_$seed$split$other"
elif [[ $extra_info == true ]]; then
  prefix="$seed$split$other"
else
  prefix="no_extra_$seed$split$other"
fi

mkdir -p $out_dir/$model_name_or_path/$prefix
cp $SCRIPT_PATH $out_dir/$model_name_or_path/$prefix/$FILE_NAME

for ((userwise_index=20; userwise_index<=102; userwise_index++));
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
	echo "userwise_split  | $userwise_split"

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
	  --num_train_epochs 200\
	  --eval_steps 10\
	  --save_steps 10\
	  --evaluation_strategy steps\
	  --lr_scheduler_type constant\
	  --learning_rate 1e-6\
	  --warmup_steps 100\
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
          --userize_ctr_threshold=$userize_ctr_threshold\

done
# ===== END ==== #

# ===== Start ==== # 
userize=true
#userize_ufeat_type="spectral_cluster_centroid_news"
#userize_ufeat_type="average_news"
#userize_ufeat_type="closest_news"
#userize_ufeat_type="user_feat"
userize_ufeat_type="tail_feat"
userize_complex_proj=false
userize_type_embedding=false
userize_loss=false
userize_pum=false
userize_ctr_threshold=275 #930

userwise=true
userwise_seed=0
userwise_split="80 20 100"

split="_t80_v20_t100"
seed="seed$userwise_seed"
other="_tailfeat_$userize_ctr_threshold-NoUserloss"

if [[ $userize == true && $extra_info == true ]]; then
  prefix="userize_$seed$split$other"
elif [[ $userize == true ]]; then
  prefix="userize_no_extra_$seed$split$other"
elif [[ $extra_info == true ]]; then
  prefix="$seed$split$other"
else
  prefix="no_extra_$seed$split"
fi

mkdir -p $out_dir/$model_name_or_path/$prefix
cp $SCRIPT_PATH $out_dir/$model_name_or_path/$prefix/$FILE_NAME

for ((userwise_index=20; userwise_index<=102; userwise_index++));
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
	echo "userwise_split  | $userwise_split"

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
	  --num_train_epochs 200\
	  --eval_steps 10\
	  --save_steps 10\
	  --evaluation_strategy steps\
	  --lr_scheduler_type constant\
	  --learning_rate 1e-6\
	  --warmup_steps 100\
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
          --userize_ctr_threshold=$userize_ctr_threshold\

done
# ===== END ==== #
# ===== Start ==== # 
userize=true
#userize_ufeat_type="spectral_cluster_centroid_news"
#userize_ufeat_type="average_news"
#userize_ufeat_type="closest_news"
#userize_ufeat_type="user_feat"
userize_ufeat_type="tail_feat"
userize_complex_proj=false
userize_type_embedding=false
userize_loss=false
userize_pum=false
userize_ctr_threshold=930

userwise=true
userwise_seed=0
userwise_split="80 20 100"

split="_t80_v20_t100"
seed="seed$userwise_seed"
other="_tailfeat_$userize_ctr_threshold-NoUserloss"

if [[ $userize == true && $extra_info == true ]]; then
  prefix="userize_$seed$split$other"
elif [[ $userize == true ]]; then
  prefix="userize_no_extra_$seed$split$other"
elif [[ $extra_info == true ]]; then
  prefix="$seed$split$other"
else
  prefix="no_extra_$seed$split$other"
fi

mkdir -p $out_dir/$model_name_or_path/$prefix
cp $SCRIPT_PATH $out_dir/$model_name_or_path/$prefix/$FILE_NAME

for ((userwise_index=20; userwise_index<=102; userwise_index++));
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
	echo "userwise_split  | $userwise_split"

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
	  --num_train_epochs 200\
	  --eval_steps 10\
	  --save_steps 10\
	  --evaluation_strategy steps\
	  --lr_scheduler_type constant\
	  --learning_rate 1e-6\
	  --warmup_steps 100\
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
          --userize_ctr_threshold=$userize_ctr_threshold\

done
# ===== END ==== #

# ===== Start ==== # 
userize=true
#userize_ufeat_type="spectral_cluster_centroid_news"
#userize_ufeat_type="average_news"
userize_ufeat_type="closest_news"
#userize_ufeat_type="user_feat"
#userize_ufeat_type="tail_feat"
userize_complex_proj=false
userize_type_embedding=false
userize_loss=false
userize_pum=false
#userize_ctr_threshold=20 #37 #215 #275 #930

userwise=true
userwise_seed=0
userwise_split="80 20 100"

split="_t80_v20_t100"
seed="seed$userwise_seed"
other="_closest_NoUserloss"

if [[ $userize == true && $extra_info == true ]]; then
  prefix="userize_$seed$split$other"
elif [[ $userize == true ]]; then
  prefix="userize_no_extra_$seed$split$other"
elif [[ $extra_info == true ]]; then
  prefix="$seed$split$other"
else
  prefix="no_extra_$seed$split$other"
fi

mkdir -p $out_dir/$model_name_or_path/$prefix
cp $SCRIPT_PATH $out_dir/$model_name_or_path/$prefix/$FILE_NAME

for ((userwise_index=20; userwise_index<=102; userwise_index++));
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
	echo "userwise_split  | $userwise_split"

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
	  --num_train_epochs 200\
	  --eval_steps 10\
	  --save_steps 10\
	  --evaluation_strategy steps\
	  --lr_scheduler_type constant\
	  --learning_rate 1e-6\
	  --warmup_steps 100\
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
          --userize_ctr_threshold=$userize_ctr_threshold\

done
# ===== END ==== #

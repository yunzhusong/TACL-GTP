export CUDA_VISIBLE_DEVICES=2

#### Get script path and name
SCRIPT_PATH=$(readlink -f "$0")
FILE_NAME="$(basename $SCRIPT_PATH)"

dataset="s2_shared_own"
model_name_or_path="facebook/bart-base"
userize_ufeat_path="../datasets/user_feat/REC/checkpoint-47500"
predict_txt_file="../results/pens_ghg_own/bart/checkpoint-98000/all_news/generated_predictions.txt"

output_dir="../results/$dataset"

#exp_name="bart_input_pred"
exp_name="bart_input_pred_A_userize_mum_loss"
#exp_name="bart_input_pred_A_extra_online"
#exp_name="bart_input_pred_A_extra_online_A_userize_mum_loss"
#exp_name="bart_input_pred_A_extra_online_A_userize_closest_mum_loss"
#exp_name="bart_input_pred_A_extra_online_A_cpxtype_userize_average_mum_loss"
#exp_name="bart_input_pred_A_extra_online_A_cpxtype_userize_spectral_mum_loss"
#exp_name="bart_input_pred_A_extra_online_A_cpxtype_userize_kmeans_mum_loss"
#exp_name="bart_input_pred_A_userize_spectral_centroid_mum_loss_v3"
#exp_name="bart_input_pred_A_extra_online_A_userize_spectral_centroid_mum_loss_v3"
#exp_name="bart_input_pred_A_extra_online_A_userize_average_mum_loss_v3"
#exp_name="bart_input_pred_A_extra_online_A_type_userize_average_mum_sum_loss_v3"
#exp_name="bart_input_pred_A_extra_online_A_userize_spectral_mum_loss_v3"
#exp_name="bart_input_pred_A_extra_online_A_cpx_userize_closest_mum_loss_v2"
#exp_name="bart_input_pred_A_extra_online_A_userize_average_ptum_loss"
#exp_name="bart_input_pred_A_extra_online_A_userize_closest_ptum_loss"
#exp_name="bart_input_pred_A_extra_online_A_userize_tailfeat_ptum_loss"
#exp_name="bart_input_pred_A_extra_online_A_userize_textclosest"
#exp_name="bart_input_pred_A_extra_online_A_userize_average_mum_loss_v4"
#exp_name="bart_input_pred_A_extra_online_A_userize_tailfeat_930_mum_loss_v4"
#exp_name="bart_input_pred_A_extra_online_A_userize_tailfeat_dot_loss_v4"
#exp_name="bart_input_pred_A_extra_online_A_userize_tailfeat_mum_dot_loss_v4"
#exp_name="bart_input_pred_A_extra_online_A_userize_average_mum_dot_loss_v4"
#exp_name="bart_input_pred_A_extra_online_A_userize_average_v4" # NOTE: without MUM loss

# v2: correct user feature
# v3: correct mum
# v4: no grad user_proj


# NOTE:
# if userize_ufeat_type is "text_closest"
# remember to set the userize_user_token_length to a proper length

##echo "===== Start Pretraining ====="
#max_source_length=224 # 256 or 224
#max_target_length=128
#
## 64
#batch_size=28
#accumulate=2 
#extra_info=false
#retrieve_online=true
#userize=true
#userize_ufeat_type=average_news
##userize_ufeat_type=tail_feat
#userize_ctr_threshold=930 #37 #215 #275 #930
#userize_user_token_length=5
#
#echo "Running $FILE_NAME"
#echo "Experiment Name: $exp_name"
#echo "extra_info: $extra_info"
#echo "userize: $userize"
#
#mkdir -p $output_dir/$exp_name
#cp $SCRIPT_PATH $output_dir/$exp_name/$FILE_NAME
##
#### For 2nd stage pre-training
#python main.py\
#  --find_best_checkpoint true\
#  --dataset_name $dataset\
#  --max_source_length $max_source_length\
#  --max_target_length $max_target_length\
#  --generation_max_length $max_target_length\
#  --pad_to_max_length true\
#  --per_device_train_batch_size $batch_size\
#  --per_device_eval_batch_size $batch_size\
#  --gradient_accumulation_steps $accumulate\
#  --num_train_epochs 100\
#  --save_steps 500\
#  --lr_scheduler_type constant\
#  --learning_rate 1e-6\
#  --save_strategy steps\
#  --save_total_limit 1\
#  --save_model_accord_to_rouge true\
#  --predict_with_generate \
#  --report_to tensorboard\
#  --logging_steps 40\
#  --overwrite_output_dir true\
#  --overwrite_cache true\
#  --remove_unused_columns false\
#  --model_name_or_path $model_name_or_path\
#  --output_dir $output_dir/$exp_name\
#  --logging_dir $output_dir/$exp_name/log\
#  --userize $userize\
#  --userize_ufeat_path $userize_ufeat_path\
#  --userize_user_token_length $userize_user_token_length\
#  --userize_ufeat_type $userize_ufeat_type\
#  --userize_loss true\
#  --userize_mum true\
#  --userize_ctr_threshold $userize_ctr_threshold\
#  --extra_info $extra_info\
#  --retrieve_online $retrieve_online\
#  --text_file $predict_txt_file\
#  --test_file ../datasets/specialize_own/test.pkl\
#  --do_train true\
#  --do_predict false\
#  --do_eval true\
#  --evaluation_strategy steps\
#  --eval_steps 500\
#  --max_eval_samples 500\
#  --userize_dot false\
#  #--max_train_samples 100\
#  #--preprocessing_num_workers 1\

echo "===== Start Evaluation ====="

#### Start ####
#output_dir="../results/specialize_own"
#exp_name="bart_input_pred_A_extra_online_A_userize_mum_loss/checkpoint-2600"
#exp_name="bart_input_pred_small_LR_5e-7/checkpoint-4600"
#exp_name="bart_input_pred_A_extra_online_A_userize_average_v4/checkpoint-8000"
#exp_name="bart_input_pred_A_extra_online_A_userize_mum_loss_run3/checkpoint-4500"

output_dir="../results/s2_shared_own"
exp_name="bart_input_pred_A_extra_online_A_userize_average_mum_loss_v4/checkpoint-8500"

max_source_length=224 # 256 or 224
max_target_length=128
extra_info=true
retrieve_online=true
userize=true
#userize_ufeat_type="average_news"
#userize_ufeat_type="closest_news"
#userize_ufeat_type="user_feat"
#userize_ufeat_type="text_closest"
userize_user_token_length=5
userize_loss=false
userize_mum=false

userize_ufeat_type="tail_feat"
userize_ctr_threshold=20 #37 #215 #275 #930
userize_ctr_quant=0.1

# For evaluation
python main.py\
  --do_predict true\
  --eval_with_test_data true\
  --test_file ../datasets/specialize_own/test.pkl\
  --dataset_name $dataset\
  --max_source_length $max_source_length\
  --max_target_length $max_target_length\
  --generation_max_length $max_target_length\
  --pad_to_max_length true\
  --per_device_eval_batch_size 128\
  --predict_with_generate true\
  --overwrite_output_dir true\
  --overwrite_cache true\
  --remove_unused_columns false\
  --model_name_or_path $output_dir/$exp_name \
  --output_dir $output_dir/$exp_name/tailfeat/$userize_ctr_quant \
  --corrupt false\
  --corrupt_token_infilling false\
  --corrupt_max_mask_num 3\
  --corrupt_max_mask_length_lambda 3\
  --corrupt_token_shuffle false\
  --userize $userize\
  --userize_ufeat_path $userize_ufeat_path\
  --userize_user_token_length $userize_user_token_length\
  --userize_ufeat_type $userize_ufeat_type\
  --userize_loss $userize_loss\
  --userize_mum $userize_mum\
  --userize_dot true\
  --extra_info $extra_info\
  --retrieve_online true\
  --text_file $predict_txt_file\
  --userize_ctr_quant $userize_ctr_quant\

### End ####
userize_ctr_quant=0.2

# For evaluation
python main.py\
  --do_predict true\
  --eval_with_test_data true\
  --test_file ../datasets/specialize_own/test.pkl\
  --dataset_name $dataset\
  --max_source_length $max_source_length\
  --max_target_length $max_target_length\
  --generation_max_length $max_target_length\
  --pad_to_max_length true\
  --per_device_eval_batch_size 128\
  --predict_with_generate true\
  --overwrite_output_dir true\
  --overwrite_cache true\
  --remove_unused_columns false\
  --model_name_or_path $output_dir/$exp_name\
  --output_dir $output_dir/$exp_name/tailfeat_$userize_ctr_quant\
  --corrupt false\
  --corrupt_token_infilling false\
  --corrupt_max_mask_num 3\
  --corrupt_max_mask_length_lambda 3\
  --corrupt_token_shuffle false\
  --userize $userize\
  --userize_ufeat_path $userize_ufeat_path\
  --userize_user_token_length $userize_user_token_length\
  --userize_ufeat_type $userize_ufeat_type\
  --userize_loss $userize_loss\
  --userize_mum $userize_mum\
  --userize_dot true\
  --extra_info $extra_info\
  --retrieve_online true\
  --text_file $predict_txt_file\
  --userize_ctr_quant $userize_ctr_quant\

### End ####
userize_ctr_quant=0.3

# For evaluation
python main.py\
  --do_predict true\
  --eval_with_test_data true\
  --test_file ../datasets/specialize_own/test.pkl\
  --dataset_name $dataset\
  --max_source_length $max_source_length\
  --max_target_length $max_target_length\
  --generation_max_length $max_target_length\
  --pad_to_max_length true\
  --per_device_eval_batch_size 128\
  --predict_with_generate true\
  --overwrite_output_dir true\
  --overwrite_cache true\
  --remove_unused_columns false\
  --model_name_or_path $output_dir/$exp_name\
  --output_dir $output_dir/$exp_name/tailfeat_$userize_ctr_quant\
  --corrupt false\
  --corrupt_token_infilling false\
  --corrupt_max_mask_num 3\
  --corrupt_max_mask_length_lambda 3\
  --corrupt_token_shuffle false\
  --userize $userize\
  --userize_ufeat_path $userize_ufeat_path\
  --userize_user_token_length $userize_user_token_length\
  --userize_ufeat_type $userize_ufeat_type\
  --userize_loss $userize_loss\
  --userize_mum $userize_mum\
  --userize_dot true\
  --extra_info $extra_info\
  --retrieve_online true\
  --text_file $predict_txt_file\
  --userize_ctr_quant $userize_ctr_quant\

### End ####
userize_ctr_quant=0.4

# For evaluation
python main.py\
  --do_predict true\
  --eval_with_test_data true\
  --test_file ../datasets/specialize_own/test.pkl\
  --dataset_name $dataset\
  --max_source_length $max_source_length\
  --max_target_length $max_target_length\
  --generation_max_length $max_target_length\
  --pad_to_max_length true\
  --per_device_eval_batch_size 128\
  --predict_with_generate true\
  --overwrite_output_dir true\
  --overwrite_cache true\
  --remove_unused_columns false\
  --model_name_or_path $output_dir/$exp_name\
  --output_dir $output_dir/$exp_name/tailfeat_$userize_ctr_quant\
  --corrupt false\
  --corrupt_token_infilling false\
  --corrupt_max_mask_num 3\
  --corrupt_max_mask_length_lambda 3\
  --corrupt_token_shuffle false\
  --userize $userize\
  --userize_ufeat_path $userize_ufeat_path\
  --userize_user_token_length $userize_user_token_length\
  --userize_ufeat_type $userize_ufeat_type\
  --userize_loss $userize_loss\
  --userize_mum $userize_mum\
  --userize_dot true\
  --extra_info $extra_info\
  --retrieve_online true\
  --text_file $predict_txt_file\
  --userize_ctr_quant $userize_ctr_quant\

### End ####
userize_ctr_quant=0.5

# For evaluation
python main.py\
  --do_predict true\
  --eval_with_test_data true\
  --test_file ../datasets/specialize_own/test.pkl\
  --dataset_name $dataset\
  --max_source_length $max_source_length\
  --max_target_length $max_target_length\
  --generation_max_length $max_target_length\
  --pad_to_max_length true\
  --per_device_eval_batch_size 128\
  --predict_with_generate true\
  --overwrite_output_dir true\
  --overwrite_cache true\
  --remove_unused_columns false\
  --model_name_or_path $output_dir/$exp_name\
  --output_dir $output_dir/$exp_name/tailfeat_$userize_ctr_quant\
  --corrupt false\
  --corrupt_token_infilling false\
  --corrupt_max_mask_num 3\
  --corrupt_max_mask_length_lambda 3\
  --corrupt_token_shuffle false\
  --userize $userize\
  --userize_ufeat_path $userize_ufeat_path\
  --userize_user_token_length $userize_user_token_length\
  --userize_ufeat_type $userize_ufeat_type\
  --userize_loss $userize_loss\
  --userize_mum $userize_mum\
  --userize_dot true\
  --extra_info $extra_info\
  --retrieve_online true\
  --text_file $predict_txt_file\
  --userize_ctr_quant $userize_ctr_quant\

### End ####
userize_ctr_quant=0.6

# For evaluation
python main.py\
  --do_predict true\
  --eval_with_test_data true\
  --test_file ../datasets/specialize_own/test.pkl\
  --dataset_name $dataset\
  --max_source_length $max_source_length\
  --max_target_length $max_target_length\
  --generation_max_length $max_target_length\
  --pad_to_max_length true\
  --per_device_eval_batch_size 128\
  --predict_with_generate true\
  --overwrite_output_dir true\
  --overwrite_cache true\
  --remove_unused_columns false\
  --model_name_or_path $output_dir/$exp_name\
  --output_dir $output_dir/$exp_name/tailfeat_$userize_ctr_quant\
  --corrupt false\
  --corrupt_token_infilling false\
  --corrupt_max_mask_num 3\
  --corrupt_max_mask_length_lambda 3\
  --corrupt_token_shuffle false\
  --userize $userize\
  --userize_ufeat_path $userize_ufeat_path\
  --userize_user_token_length $userize_user_token_length\
  --userize_ufeat_type $userize_ufeat_type\
  --userize_loss $userize_loss\
  --userize_mum $userize_mum\
  --userize_dot true\
  --extra_info $extra_info\
  --retrieve_online true\
  --text_file $predict_txt_file\
  --userize_ctr_quant $userize_ctr_quant\

### End ####
userize_ctr_quant=0.7

# For evaluation
python main.py\
  --do_predict true\
  --eval_with_test_data true\
  --test_file ../datasets/specialize_own/test.pkl\
  --dataset_name $dataset\
  --max_source_length $max_source_length\
  --max_target_length $max_target_length\
  --generation_max_length $max_target_length\
  --pad_to_max_length true\
  --per_device_eval_batch_size 128\
  --predict_with_generate true\
  --overwrite_output_dir true\
  --overwrite_cache true\
  --remove_unused_columns false\
  --model_name_or_path $output_dir/$exp_name\
  --output_dir $output_dir/$exp_name/tailfeat_$userize_ctr_quant\
  --corrupt false\
  --corrupt_token_infilling false\
  --corrupt_max_mask_num 3\
  --corrupt_max_mask_length_lambda 3\
  --corrupt_token_shuffle false\
  --userize $userize\
  --userize_ufeat_path $userize_ufeat_path\
  --userize_user_token_length $userize_user_token_length\
  --userize_ufeat_type $userize_ufeat_type\
  --userize_loss $userize_loss\
  --userize_mum $userize_mum\
  --userize_dot true\
  --extra_info $extra_info\
  --retrieve_online true\
  --text_file $predict_txt_file\
  --userize_ctr_quant $userize_ctr_quant\

### End ####
userize_ctr_quant=0.8

# For evaluation
python main.py\
  --do_predict true\
  --eval_with_test_data true\
  --test_file ../datasets/specialize_own/test.pkl\
  --dataset_name $dataset\
  --max_source_length $max_source_length\
  --max_target_length $max_target_length\
  --generation_max_length $max_target_length\
  --pad_to_max_length true\
  --per_device_eval_batch_size 128\
  --predict_with_generate true\
  --overwrite_output_dir true\
  --overwrite_cache true\
  --remove_unused_columns false\
  --model_name_or_path $output_dir/$exp_name\
  --output_dir $output_dir/$exp_name/tailfeat_$userize_ctr_quant\
  --corrupt false\
  --corrupt_token_infilling false\
  --corrupt_max_mask_num 3\
  --corrupt_max_mask_length_lambda 3\
  --corrupt_token_shuffle false\
  --userize $userize\
  --userize_ufeat_path $userize_ufeat_path\
  --userize_user_token_length $userize_user_token_length\
  --userize_ufeat_type $userize_ufeat_type\
  --userize_loss $userize_loss\
  --userize_mum $userize_mum\
  --userize_dot true\
  --extra_info $extra_info\
  --retrieve_online true\
  --text_file $predict_txt_file\
  --userize_ctr_quant $userize_ctr_quant\

### End ####
userize_ctr_quant=0.9

# For evaluation
python main.py\
  --do_predict true\
  --eval_with_test_data true\
  --test_file ../datasets/specialize_own/test.pkl\
  --dataset_name $dataset\
  --max_source_length $max_source_length\
  --max_target_length $max_target_length\
  --generation_max_length $max_target_length\
  --pad_to_max_length true\
  --per_device_eval_batch_size 128\
  --predict_with_generate true\
  --overwrite_output_dir true\
  --overwrite_cache true\
  --remove_unused_columns false\
  --model_name_or_path $output_dir/$exp_name\
  --output_dir $output_dir/$exp_name/tailfeat_$userize_ctr_quant\
  --corrupt false\
  --corrupt_token_infilling false\
  --corrupt_max_mask_num 3\
  --corrupt_max_mask_length_lambda 3\
  --corrupt_token_shuffle false\
  --userize $userize\
  --userize_ufeat_path $userize_ufeat_path\
  --userize_user_token_length $userize_user_token_length\
  --userize_ufeat_type $userize_ufeat_type\
  --userize_loss $userize_loss\
  --userize_mum $userize_mum\
  --userize_dot true\
  --extra_info $extra_info\
  --retrieve_online true\
  --text_file $predict_txt_file\
  --userize_ctr_quant $userize_ctr_quant\

### End ####

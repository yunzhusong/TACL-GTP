export CUDA_VISIBLE_DEVICES=1

#### Get script path and name
SCRIPT_PATH=$(readlink -f "$0")
FILE_NAME="$(basename $SCRIPT_PATH)"

dataset="s2_shared_own"
model_name_or_path="facebook/bart-base"
#model_name_or_path="../results/pens_ghg_own/bart/checkpoint-98000"
userize_ufeat_path="../datasets/user_feat/REC/checkpoint-47500"

output_dir="../results/$dataset"
#exp_name="develop"
#exp_name="bart_input_pred"
#exp_name="bart_input_pred_A_userize_mum_loss"
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
#exp_name="bart_input_pred_A_extra_online_A_type_userize_average_pum_loss"
#exp_name="bart_input_pred_A_extra_online_A_userize_average_ptum_loss"
#exp_name="bart_input_pred_A_extra_online_A_userize_closest_ptum_loss"
#exp_name="init_ghg/bart_input_pred_A_extra_online_A_userize_closest_ptum_loss"
#exp_name="init_ghg/bart_input_pred_A_extra_online_A_userize_userfeat_ptum_loss"
#exp_name="bart_input_pred_A_extra_online_A_userize_tailfeat_ptum_loss"
#exp_name="bart_input_pred_A_extra_online_A_userize_textclosest"
#exp_name="bart_input_pred_A_extra_online_A_userize_average_mum_loss_v4"
#exp_name="bart_input_pred_A_extra_online_A_userize_tailfeat_mum_loss_v4"
#exp_name="bart_input_pred_A_extra_online_A_userize_tailfeat_dot_loss_v4"
#exp_name="bart_input_pred_A_extra_online_A_userize_tailfeat_mum_dot_loss_v4"
#exp_name="bart_input_pred_A_extra_online_A_userize_average_mum_dot_loss_v4"
exp_name="bart_userize_average_v4"

# v2: correct user feature
# v3: correct mum
# v4: no grad user_proj


# NOTE:
# if userize_ufeat_type is "text_closest"
# remember to set the userize_user_token_length to a proper length
# and turn off all user loss (mum and pum)

#echo "===== Start Pretraining ====="
#max_source_length=512 # 256 or 224
#max_target_length=64
#
## 64
#batch_size=12
#accumulate=4
#userize=true
#userize_ufeat_type=average_news
##userize_ufeat_type=tail_feat
#userize_ctr_threshold=20 #37 #215 #275 #930
#userize_user_token_length=5
#
#echo "Running $FILE_NAME"
#echo "Experiment Name: $exp_name"
#echo "userize: $userize"
#
#mkdir -p $output_dir/$exp_name
#cp $SCRIPT_PATH $output_dir/$exp_name/$FILE_NAME
#
## For single-stage training
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
#  --save_steps 1000\
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
#  --userize_loss false\
#  --userize_mum false\
#  --userize_dot false\
#  --test_file ../datasets/specialize_own/test.pkl\
#  --do_train true\
#  --do_predict false\
#  --do_eval true\
#  --evaluation_strategy steps\
#  --eval_steps 1000\
#  --max_eval_samples 500\
#  --text_column body\
#  #--max_train_samples 100\
#  #--preprocessing_num_workers 1\
#
#echo "===== Start Evaluation ====="
#
###### Start ####
#exp_name="bart_input_pred_A_extra_online_A_userize_average_dot_loss_v4/checkpoint-7500"
exp_name="bart_userize_average_v4/checkpoint-13000"
max_source_length=224 # 256 or 224
max_target_length=128
userize=true
userize_ufeat_type="average_news"
#userize_ufeat_type="closest_news"
#userize_ufeat_type="user_feat"
#userize_ufeat_type="text_closest"
userize_user_token_length=5
userize_loss=false
userize_pum=false
userize_mum=false

#userize_ufeat_type="tail_feat"
userize_ctr_threshold=20 #37 #215 #275 #930

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
  --output_dir $output_dir/$exp_name\
  --userize $userize\
  --userize_ufeat_path $userize_ufeat_path\
  --userize_user_token_length $userize_user_token_length\
  --userize_ufeat_type $userize_ufeat_type\
  --userize_loss $userize_loss\
  --userize_mum $userize_mum\
  --userize_pum $userize_pum\
  --userize_dot false\
  --text_column body\

### End ####

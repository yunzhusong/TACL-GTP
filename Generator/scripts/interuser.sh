export CUDA_VISIBLE_DEVICES=1

#### Get script path and name
SCRIPT_PATH=$(readlink -f "$0")
FILE_NAME="$(basename $SCRIPT_PATH)"

dataset="s2_shared_own"
model_dir="../results_important/s2_shared_own"
out_dir="../results_important/s2_shared_own_userize"
log_dir="../results_important/s2_shared_own_userize/logging"

userize_ufeat_path="../datasets_old/user_feat/REC/checkpoint-47500"
predict_txt_file="../results_important/pens_ghg_own/bart/checkpoint-98000/all_news/generated_predictions.txt"

# Two-stage
#model_name_or_path="bart-base"
# GTP
model_name_or_path="bart_input_pred_A_extra_online_A_userize_average_mum_loss_v4/checkpoint-8500"

train_epoch=100
learning_rate=6e-6

max_source_length=256 # 256 or 224
max_target_length=64
batch_size=16
accumulate=1

extra_info=false
retrieve_online=true

userize=true
userize_ufeat_type="user_feat"
userize_ctr_threshold=20 #37 #215 #275 #930

#exp_name="seed1_gtp_run6"
#setting="INTERUSER/50_3_50"
#mkdir -p $out_dir/$setting/$model_name_or_path/$exp_name
#
#python main.py\
#  --do_finetune true\
#  --do_train true\
#  --do_eval true\
#  --do_predict true\
#  --find_best_checkpoint true\
#  --remove_after_predict true\
#  --dataset_name $dataset\
#  --max_source_length $max_source_length\
#  --max_target_length $max_target_length\
#  --generation_max_length $max_target_length\
#  --pad_to_max_length true\
#  --per_device_train_batch_size $batch_size\
#  --per_device_eval_batch_size $batch_size\
#  --gradient_accumulation_steps $accumulate\
#  --num_train_epochs $train_epoch\
#  --eval_steps 1000\
#  --save_steps 1000\
#  --evaluation_strategy steps\
#  --lr_scheduler_type linear\
#  --learning_rate $learning_rate\
#  --warmup_steps 5000\
#  --save_strategy steps\
#  --save_total_limit 1\
#  --save_model_accord_to_rouge true\
#  --predict_with_generate true\
#  --report_to tensorboard\
#  --logging_steps 100\
#  --overwrite_output_dir true\
#  --overwrite_cache true\
#  --remove_unused_columns false\
#  --model_name_or_path $model_dir/$model_name_or_path\
#  --output_dir $out_dir/$setting/$model_name_or_path/$exp_name\
#  --logging_dir $log_dir/$setting/$model_name_or_path/$exp_name/log\
#  --text_file $predict_txt_file\
#  --userize $userize\
#  --userize_ufeat_path $userize_ufeat_path\
#  --userize_user_token_length 5\
#  --userize_loss false\
#  --userize_mum false\
#  --userize_ufeat_type $userize_ufeat_type\
#  --extra_info $extra_info\
#  --retrieve_online $retrieve_online\
#  --userwise true\
#  --userwise_index -1\
#  --userwise_split "10000 600 10000"\
#  --userwise_seed 1\
#  --userwise_sample_type inter_user\
#  --userize_ctr_threshold=$userize_ctr_threshold\
#  --developing true\
#
#model_name_or_path="bart_input_pred_A_extra_online_A_userize_average_mum_loss_v4/checkpoint-8500"
#exp_name="seed10_gtp_run6"
#setting="INTERUSER/50_3_50"
#mkdir -p $out_dir/$setting/$model_name_or_path/$exp_name
#
#python main.py\
#  --do_finetune true\
#  --do_train true\
#  --do_eval true\
#  --do_predict true\
#  --find_best_checkpoint true\
#  --remove_after_predict true\
#  --dataset_name $dataset\
#  --max_source_length $max_source_length\
#  --max_target_length $max_target_length\
#  --generation_max_length $max_target_length\
#  --pad_to_max_length true\
#  --per_device_train_batch_size $batch_size\
#  --per_device_eval_batch_size $batch_size\
#  --gradient_accumulation_steps $accumulate\
#  --num_train_epochs $train_epoch\
#  --eval_steps 1000\
#  --save_steps 1000\
#  --evaluation_strategy steps\
#  --lr_scheduler_type linear\
#  --learning_rate $learning_rate\
#  --warmup_steps 5000\
#  --save_strategy steps\
#  --save_total_limit 1\
#  --save_model_accord_to_rouge true\
#  --predict_with_generate true\
#  --report_to tensorboard\
#  --logging_steps 100\
#  --overwrite_output_dir true\
#  --overwrite_cache true\
#  --remove_unused_columns false\
#  --model_name_or_path $model_dir/$model_name_or_path\
#  --output_dir $out_dir/$setting/$model_name_or_path/$exp_name\
#  --logging_dir $log_dir/$setting/$model_name_or_path/$exp_name/log\
#  --text_file $predict_txt_file\
#  --userize $userize\
#  --userize_ufeat_path $userize_ufeat_path\
#  --userize_user_token_length 5\
#  --userize_loss false\
#  --userize_mum false\
#  --userize_ufeat_type $userize_ufeat_type\
#  --extra_info $extra_info\
#  --retrieve_online $retrieve_online\
#  --userwise true\
#  --userwise_index -1\
#  --userwise_split "10000 600 10000"\
#  --userwise_seed 10\
#  --userwise_sample_type inter_user\
#  --userize_ctr_threshold=$userize_ctr_threshold\
#  --developing true\

##
##
model_name_or_path="bart-base"
exp_name="seed1_two_stage_userize_run2"
setting="INTERUSER/50_3_50"
mkdir -p $out_dir/$setting/$model_name_or_path/$exp_name

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
  --num_train_epochs $train_epoch\
  --eval_steps 1000\
  --save_steps 1000\
  --evaluation_strategy steps\
  --lr_scheduler_type linear\
  --learning_rate $learning_rate\
  --warmup_steps 5000\
  --save_strategy steps\
  --save_total_limit 1\
  --save_model_accord_to_rouge true\
  --predict_with_generate true\
  --report_to tensorboard\
  --logging_steps 100\
  --overwrite_output_dir true\
  --overwrite_cache true\
  --remove_unused_columns false\
  --model_name_or_path $model_dir/$model_name_or_path\
  --output_dir $out_dir/$setting/$model_name_or_path/$exp_name\
  --logging_dir $log_dir/$setting/$model_name_or_path/$exp_name/log\
  --text_file $predict_txt_file\
  --userize $userize\
  --userize_ufeat_path $userize_ufeat_path\
  --userize_user_token_length 5\
  --userize_loss false\
  --userize_mum false\
  --userize_ufeat_type $userize_ufeat_type\
  --extra_info $extra_info\
  --retrieve_online $retrieve_online\
  --userwise true\
  --userwise_index -1\
  --userwise_split "10000 600 10000"\
  --userwise_seed 1\
  --userwise_sample_type inter_user\
  --userize_ctr_threshold=$userize_ctr_threshold\
  --developing true\

#model_name_or_path="bart-base"
#exp_name="seed10_two_stage_userize"
#setting="INTERUSER/50_3_50"
#mkdir -p $out_dir/$setting/$model_name_or_path/$exp_name
#
#python main.py\
#  --do_finetune true\
#  --do_train true\
#  --do_eval true\
#  --do_predict true\
#  --find_best_checkpoint true\
#  --remove_after_predict true\
#  --dataset_name $dataset\
#  --max_source_length $max_source_length\
#  --max_target_length $max_target_length\
#  --generation_max_length $max_target_length\
#  --pad_to_max_length true\
#  --per_device_train_batch_size $batch_size\
#  --per_device_eval_batch_size $batch_size\
#  --gradient_accumulation_steps $accumulate\
#  --num_train_epochs $train_epoch\
#  --eval_steps 1000\
#  --save_steps 1000\
#  --evaluation_strategy steps\
#  --lr_scheduler_type linear\
#  --learning_rate $learning_rate\
#  --warmup_steps 5000\
#  --save_strategy steps\
#  --save_total_limit 1\
#  --save_model_accord_to_rouge true\
#  --predict_with_generate true\
#  --report_to tensorboard\
#  --logging_steps 100\
#  --overwrite_output_dir true\
#  --overwrite_cache true\
#  --remove_unused_columns false\
#  --model_name_or_path $model_dir/$model_name_or_path\
#  --output_dir $out_dir/$setting/$model_name_or_path/$exp_name\
#  --logging_dir $log_dir/$setting/$model_name_or_path/$exp_name/log\
#  --text_file $predict_txt_file\
#  --userize $userize\
#  --userize_ufeat_path $userize_ufeat_path\
#  --userize_user_token_length 5\
#  --userize_loss false\
#  --userize_mum false\
#  --userize_ufeat_type $userize_ufeat_type\
#  --extra_info $extra_info\
#  --retrieve_online $retrieve_online\
#  --userwise true\
#  --userwise_index -1\
#  --userwise_split "10000 600 10000"\
#  --userwise_seed 10\
#  --userwise_sample_type inter_user\
#  --userize_ctr_threshold=$userize_ctr_threshold\
#  --developing true\

## two_stage without userize
#model_name_or_path="bart-base"
#exp_name="seed1_two_stage_wo_userize"
#setting="INTERUSER/50_3_50"
#mkdir -p $out_dir/$setting/$model_name_or_path/$exp_name
#
#python main.py\
#  --do_finetune true\
#  --do_train true\
#  --do_eval true\
#  --do_predict true\
#  --find_best_checkpoint true\
#  --remove_after_predict true\
#  --dataset_name $dataset\
#  --max_source_length $max_source_length\
#  --max_target_length $max_target_length\
#  --generation_max_length $max_target_length\
#  --pad_to_max_length true\
#  --per_device_train_batch_size $batch_size\
#  --per_device_eval_batch_size $batch_size\
#  --gradient_accumulation_steps $accumulate\
#  --num_train_epochs $train_epoch\
#  --eval_steps 1000\
#  --save_steps 1000\
#  --evaluation_strategy steps\
#  --lr_scheduler_type linear\
#  --learning_rate $learning_rate\
#  --warmup_steps 5000\
#  --save_strategy steps\
#  --save_total_limit 1\
#  --save_model_accord_to_rouge true\
#  --predict_with_generate true\
#  --report_to tensorboard\
#  --logging_steps 100\
#  --overwrite_output_dir true\
#  --overwrite_cache true\
#  --remove_unused_columns false\
#  --model_name_or_path $model_dir/$model_name_or_path\
#  --output_dir $out_dir/$setting/$model_name_or_path/$exp_name\
#  --logging_dir $log_dir/$setting/$model_name_or_path/$exp_name/log\
#  --text_file $predict_txt_file\
#  --userize false\
#  --userize_ufeat_path $userize_ufeat_path\
#  --userize_user_token_length 5\
#  --userize_loss false\
#  --userize_mum false\
#  --userize_ufeat_type $userize_ufeat_type\
#  --extra_info $extra_info\
#  --retrieve_online $retrieve_online\
#  --userwise true\
#  --userwise_index -1\
#  --userwise_split "10000 600 10000"\
#  --userwise_seed 1\
#  --userwise_sample_type inter_user\
#  --userize_ctr_threshold=$userize_ctr_threshold\
#  --developing true\
#
#model_name_or_path="bart-base"
#exp_name="seed10_two_stage_wo_userize"
#setting="INTERUSER/50_3_50"
#mkdir -p $out_dir/$setting/$model_name_or_path/$exp_name
#
#python main.py\
#  --do_finetune true\
#  --do_train true\
#  --do_eval true\
#  --do_predict true\
#  --find_best_checkpoint true\
#  --remove_after_predict true\
#  --dataset_name $dataset\
#  --max_source_length $max_source_length\
#  --max_target_length $max_target_length\
#  --generation_max_length $max_target_length\
#  --pad_to_max_length true\
#  --per_device_train_batch_size $batch_size\
#  --per_device_eval_batch_size $batch_size\
#  --gradient_accumulation_steps $accumulate\
#  --num_train_epochs $train_epoch\
#  --eval_steps 1000\
#  --save_steps 1000\
#  --evaluation_strategy steps\
#  --lr_scheduler_type linear\
#  --learning_rate $learning_rate\
#  --warmup_steps 5000\
#  --save_strategy steps\
#  --save_total_limit 1\
#  --save_model_accord_to_rouge true\
#  --predict_with_generate true\
#  --report_to tensorboard\
#  --logging_steps 100\
#  --overwrite_output_dir true\
#  --overwrite_cache true\
#  --remove_unused_columns false\
#  --model_name_or_path $model_dir/$model_name_or_path\
#  --output_dir $out_dir/$setting/$model_name_or_path/$exp_name\
#  --logging_dir $log_dir/$setting/$model_name_or_path/$exp_name/log\
#  --text_file $predict_txt_file\
#  --userize false\
#  --userize_ufeat_path $userize_ufeat_path\
#  --userize_user_token_length 5\
#  --userize_loss false\
#  --userize_mum false\
#  --userize_ufeat_type $userize_ufeat_type\
#  --extra_info $extra_info\
#  --retrieve_online $retrieve_online\
#  --userwise true\
#  --userwise_index -1\
#  --userwise_split "10000 600 10000"\
#  --userwise_seed 10\
#  --userwise_sample_type inter_user\
#  --userize_ctr_threshold=$userize_ctr_threshold\
#  --developing true\

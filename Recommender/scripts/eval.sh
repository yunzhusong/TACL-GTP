#model_path="../results/s2_own/userize/bart_input_pred_A_extra_online_A_cpxtype_userize_average_mum_loss/checkpoint-8200/userize_seed0_t80_v20_t100"
#model_path="../results/s2_own/userize/bart_input_pred_A_extra_online_A_userize_average_mum_loss/checkpoint-9000/userize_seed0_t80_v20_t100"
#model_path="../results/s2_own/userize/bart_input_pred_A_extra_online_A_userize_closest_mum_loss/checkpoint-9000/userize_seed0_t80_v20_t100"
#model_path="../results/s2_own/bart_input_pred_A_extra_online_A_userize_average_mum_loss/checkpoint-9000/userwise_own/userize_seed0_t80_v20_t100"
#model_path="../results/s2_own/userize/bart_input_pred_A_userize_mum_loss/checkpoint-11400/userize_seed0_t80_v20_t100"

#model_path="../results/s2_own/userize/bart_input_pred_A_extra_online_A_userize_average_mum_loss_v3/checkpoint-9000/seed0_t80_v20_t100" #ape_avg_v3
#model_path="../results/s2_own/userize/bart_input_pred_A_extra_online_A_userize_spectral_centroid_mum_loss_v3/checkpoint-6800/seed0_t80_v20_t100/"
#model_path="../results/s2_own/userize/bart_input_pred_A_userize_spectral_centroid_mum_loss_v3/checkpoint-9400/seed0_t80_v20_t100"
#model_path="../results/s2_own/userize/bart_input_pred_A_extra_online/checkpoint-9000/seed0_t80_v20_t100"
#model_path="../results/s2_shared_own_userize/bart_input_pred_A_extra_online_A_userize_tailfeat_mum_dot_loss_v4/checkpoint-8000/userize_seed0_t80_v20_t100_average_NoUserloss_dev"

model_path="../results/s2_shared_own_userize/bart-base/no_extra_seed0_t80_v20_t100_dev"
out_path="../APE/datasets/test/raw/bart_base_80_20_100"

# 1. Format the prediction results
python ape/file_format.py\
  --model_path $model_path\
  --out_path $out_path\
  --num_user 10\

# 2. Run recommendation model

# 3. Add the model_path to the following file and run it
#python anal/interpret_results.py


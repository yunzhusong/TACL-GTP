# APE Evaluation
export CUDA_VISIBLE_DEVICES=1

MODEL_TYPE=transformer_bottleneck
FOLD_SIZE=100

### Prediction for User/Editor Titles ###
#TITLE_VERSION=user_title
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_VERSION
#done

#TITLE_VERSION=random200_editor_title
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_VERSION
#done

#### Zero-shot ###
#TITLE_VERSION=chatGPT
#TITLE_PATH=../results_important/chatGPT
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH
#done

#TITLE_VERSION=old_stage1
#TITLE_PATH=../results_important/pens_ghg_own/bart/checkpoint-98000/phg
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH
#done

#TITLE_VERSION=stage1
#TITLE_PATH=../results/ghg/bart/checkpoint-32000
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH
#done
#
#TITLE_VERSION=early_fusion
#TITLE_PATH=../results_important/s2_shared_own/bart_userize_average_v4/checkpoint-13000
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH
#done
#
#TITLE_VERSION=gtp_wo_penssh_old
#TITLE_PATH=../results_important/specialize_own/bart_input_pred_A_extra_online_A_userize_mum_loss/checkpoint-2600
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH
#done

#TITLE_VERSION=gtp_wo_penssh_1
#TITLE_PATH=../results_important/specialize_own/bart_input_pred_A_extra_online
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH
#done
#
#TITLE_VERSION=gtp_wo_penssh_2
#TITLE_PATH=../results_important/specialize_own/bart_input_pred_A_extra_online_A_userize
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH
#        ###
#done

#TITLE_VERSION=old_gtp
#TITLE_PATH=../results_important/s2_shared_own/bart_input_pred_A_extra_online_A_userize_average_mum_loss_v4/checkpoint-8500
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH
#        ###
#done
#
#TITLE_VERSION=new_gtp
#TITLE_PATH=../results/phg/TrRM_fixEmbed/checkpoint-84000/userize_10token_isb_rec_mum_2e-6lr_run2/checkpoint-12500
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH
#        ###
#done

#TITLE_VERSION=new_gtp_run2
#TITLE_PATH=../results/phg/TrRM_fixEmbed/checkpoint-84000/userize_10token_isb_rec_2e-6lr/checkpoint-6000
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH
#        ###
#done
#
#
#TITLE_VERSION=new_gtp_run3
#TILE_PATH=../results/phg/TrRM_fixEmbed/checkpoint-84000/userize_10token_isb_rec_xent_mum_2e-6lr/checkpoint-15500
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH
#        ###
#done
#
#### Few-Shot ####

## HG+HC
#TITLE_VERSION=seed0_hg_hc_80_20_100
#TITLE_PATH=../results_important/s2_shared_own_userize/bart-base/no_extra_seed0_t80_v20_t100_dev
#
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        multiple_files=True \
#        file_extracted=False \
#        ###
#done
#
#TITLE_VERSION=seed1_hg_hc_80_20_100
#TITLE_PATH=../results_important/s2_shared_own_userize/bart-base/t80_v20_t100_seed1/no_extra_random_dev
#
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        multiple_files=True \
#        file_extracted=False \
#        ###
#done
#
#TITLE_VERSION=seed10_hg_hc_80_20_100
#TITLE_PATH=../results_important/s2_shared_own_userize/bart-base/t80_v20_t100_seed10/no_extra_random_dev
#
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        multiple_files=True \
#        file_extracted=False \
#        ###
#done



## LF w/o pretraining
#TITLE_VERSION=seed0_lf_80_20_100
#TITLE_PATH=../results_important/s2_shared_own_userize/bart-base/userize_no_extra_seed0_t80_v20_t100_dev
#
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        multiple_files=True \
#        file_extracted=False \
#        ###
#done
#
#TITLE_VERSION=seed1_lf_80_20_100
#TITLE_PATH=../results_important/s2_shared_own_userize/bart-base/t80_v20_t100_seed1/userize_no_extra_random_average_NoUserloss_dev
#
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        multiple_files=True \
#        file_extracted=False \
#        ###
#done
#
#TITLE_VERSION=seed10_lf_80_20_100
#TITLE_PATH=../results_important/s2_shared_own_userize/bart-base/t80_v20_t100_seed10/userize_no_extra_random_average_NoUserloss_dev
#
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        multiple_files=True \
#        file_extracted=False \
#        ###
#done
#
### GTP
#TITLE_VERSION=seed0_gtp_80_20_100
#TITLE_PATH=../results_important/s2_shared_own_userize/bart_input_pred_A_extra_online_A_userize_average_mum_loss_v4/checkpoint-8500/userize_seed0_t80_v20_t100_average_NoUserloss_dev
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        multiple_files=True \
#        file_extracted=False \
#        ###
#done
#
#TITLE_VERSION=seed1_gtp_80_20_100
#TITLE_PATH=../results_important/s2_shared_own_userize/bart_input_pred_A_extra_online_A_userize_average_mum_loss_v4/checkpoint-8500/t80_v20_t100_seed1/userize_random_average_NoUserloss_dev
#
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        multiple_files=True \
#        file_extracted=False \
#        ###
#done
#
#TITLE_VERSION=seed10_gtp_80_20_100
#TITLE_PATH=../results_important/s2_shared_own_userize/bart_input_pred_A_extra_online_A_userize_average_mum_loss_v4/checkpoint-8500/t80_v20_t100_seed10/userize_random_average_NoUserloss_dev
#
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        multiple_files=True \
#        file_extracted=False \
#        ###
#done
#
## w/o TrRMIo
#TITLE_VERSION=seed0_wo_TrRMIo_80_20_100
#TITLE_PATH=../results_important/s2_shared_own_userize/bart_news_bart_input_pred_A_extra_online_A_userize_average_mum_loss_v4/checkpoint-8000/t80_v20_t100/userize_informativeness_tailfeat_275-NoUserloss_bart_news_dev
#
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        multiple_files=True \
#        file_extracted=False \
#        ###
#done
#
#TITLE_VERSION=seed1_wo_TrRMIo_80_20_100
#TITLE_PATH=../results_important/s2_shared_own_userize/bart_news_bart_input_pred_A_extra_online_A_userize_average_mum_loss_v4/checkpoint-8000/t80_v20_t100_seed1/userize_random_average_NoUserloss_dev
#
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        multiple_files=True \
#        file_extracted=False \
#        ###
#done
#
#TITLE_VERSION=seed10_wo_TrRMIo_80_20_100
#TITLE_PATH=../results_important/s2_shared_own_userize/bart_news_bart_input_pred_A_extra_online_A_userize_average_mum_loss_v4/checkpoint-8000/t80_v20_t100_seed10/userize_random_average_NoUserloss_dev
#
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        multiple_files=True \
#        file_extracted=False \
#        ###
#done
#
## w/o MUM
#TITLE_VERSION=seed0_wo_mum_80_20_100
#TITLE_PATH=../results_important/s2_shared_own_userize/bart_input_pred_A_extra_online_A_userize_average_v4/checkpoint-8000/t80_v20_t100/userize_informativeness_average_NoUserloss_dev
#
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        multiple_files=True \
#        file_extracted=False \
#        ###
#done
#
#TITLE_VERSION=seed1_wo_mum_80_20_100
#TITLE_PATH=../results_important/s2_shared_own_userize/bart_input_pred_A_extra_online_A_userize_average_v4/checkpoint-8000/t80_v20_t100_seed1/userize_random_average_NoUserloss_dev
#
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        multiple_files=True \
#        file_extracted=False \
#        ###
#done
#
#TITLE_VERSION=seed10_wo_mum_80_20_100
#TITLE_PATH=../results_important/s2_shared_own_userize/bart_input_pred_A_extra_online_A_userize_average_v4/checkpoint-8000/t80_v20_t100_seed10/userize_random_average_NoUserloss_dev
#
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        multiple_files=True \
#        file_extracted=False \
#        ###
#done
#
## w/o ISB
#TITLE_VERSION=seed0_wo_isb_80_20_100
#TITLE_PATH=../results_important/s2_shared_own_userize/bart_input_pred_A_userize_mum_loss/checkpoint-9500/t80_v20_t100/userize_no_extra_informativeness_average_NoUserloss_dev
#
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        multiple_files=True \
#        file_extracted=False \
#        ###
#done
#
#TITLE_VERSION=seed1_wo_isb_80_20_100
#TITLE_PATH=../results_important/s2_shared_own_userize/bart_input_pred_A_userize_mum_loss/checkpoint-9500/t80_v20_t100_seed1/userize_no_extra_random_average_NoUserloss_dev
#
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        multiple_files=True \
#        file_extracted=False \
#        ###
#done
#
#TITLE_VERSION=seed10_wo_isb_80_20_100
#TITLE_PATH=../results_important/s2_shared_own_userize/bart_input_pred_A_userize_mum_loss/checkpoint-9500/t80_v20_t100_seed10/userize_no_extra_random_average_NoUserloss_dev
#
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        multiple_files=True \
#        file_extracted=False \
#        ###
#done
#
## w/o pretraining



## w/o late fusion
#TITLE_VERSION=seed0_wo_lf_80_20_100
#TITLE_PATH=../results_important/s2_shared_own_userize/bart_userize_average_v4/checkpoint-13000/userize_no_extra_seed0_t80_v20_t100_average_NoUserloss_dev
#
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        multiple_files=True \
#        file_extracted=False \
#        ###
#done
#
#TITLE_VERSION=seed1_wo_lf_80_20_100
#TITLE_PATH=../results_important/s2_shared_own_userize/bart_userize_average_v4/checkpoint-13000/userize_no_extra_seed1_t80_v20_t100_seed1_average_NoUserloss_dev
#
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        multiple_files=True \
#        file_extracted=False \
#        ###
#done
#
#TITLE_VERSION=seed10_wo_lf_80_20_100
#TITLE_PATH=../results_important/s2_shared_own_userize/bart_userize_average_v4/checkpoint-13000/userize_no_extra_seed10_t80_v20_t100_seed10_average_NoUserloss_dev
#
#
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        multiple_files=True \
#        file_extracted=False \
#        ###
#done
#

#TITLE_VERSION=seed0_gtp_wo_penssh_80_20_100
#TITLE_PATH=../results_important/specialize_own_userize/bart_input_pred_A_extra_online_A_userize_mum_loss/checkpoint-2600/t80_v20_t100/userize_random_average_NoUserloss_dev
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        multiple_files=True \
#        file_extracted=False \
#        ###
#done
#


#### ANAL ####

# Random (Same as GTP)
#TITLE_VERSION=seed0_gtp_random_80_20_100
#TITLE_PATH=../results_important/s2_shared_own_userize/bart_input_pred_A_extra_online_A_userize_average_mum_loss_v4/checkpoint-8500/t80_v20_t100/userize_random_tailfeat_275-NoUserloss_dev
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        multiple_files=True \
#        file_extracted=False \
#        ###
#done
#
#TITLE_VERSION=seed1_gtp_random_80_20_100
#TITLE_PATH=../results_important/s2_shared_own_userize/bart_input_pred_A_extra_online_A_userize_average_mum_loss_v4/checkpoint-8500/t80_v20_t100_seed1/userize_random_average_NoUserloss_dev
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        multiple_files=True \
#        file_extracted=False \
#        ###
#done
#
#TITLE_VERSION=seed10_gtp_random_80_20_100
#TITLE_PATH=../results_important/s2_shared_own_userize/bart_input_pred_A_extra_online_A_userize_average_mum_loss_v4/checkpoint-8500/t80_v20_t100_seed10/userize_random_average_NoUserloss_dev
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        multiple_files=True \
#        file_extracted=False \
#        ###
#done
#
#
## Diversity
#TITLE_VERSION=seed0_gtp_diversity_80_20_100
#TITLE_PATH="../results_important/s2_shared_own_userize/bart_input_pred_A_extra_online_A_userize_average_mum_loss_v4/checkpoint-8500/t80_v20_t100/userize_diversity_average_NoUserloss_dev"
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        multiple_files=True \
#        file_extracted=False \
#        ###
#done
#
#TITLE_VERSION=seed1_gtp_diversity_80_20_100
#TITLE_PATH="../results_important/s2_shared_own_userize/bart_input_pred_A_extra_online_A_userize_average_mum_loss_v4/checkpoint-8500/t80_v20_t100_seed1/userize_diversity_tailfeat_0.1\_NoUserloss_dev"
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        multiple_files=True \
#        file_extracted=False \
#        ###
#done
#
#TITLE_VERSION=seed10_gtp_diversity_80_20_100
#TITLE_PATH="../results_important/s2_shared_own_userize/bart_input_pred_A_extra_online_A_userize_average_mum_loss_v4/checkpoint-8500/t80_v20_t100_seed10/userize_diversity_tailfeat_0.1\_NoUserloss_dev"
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        multiple_files=True \
#        file_extracted=False \
#        ###
#done
#
#
## Informativeness
#TITLE_VERSION=seed0_gtp_informativeness_80_20_100
#TITLE_PATH="../results_important/s2_shared_own_userize/bart_input_pred_A_extra_online_A_userize_average_mum_loss_v4/checkpoint-8500/t80_v20_t100/userize_informativeness_average_NoUserloss_dev"
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        multiple_files=True \
#        file_extracted=False \
#        ###
#done
#
#TITLE_VERSION=seed1_gtp_informativeness_80_20_100
#TITLE_PATH="../results_important/s2_shared_own_userize/bart_input_pred_A_extra_online_A_userize_average_mum_loss_v4/checkpoint-8500/t80_v20_t100_seed1/userize_informativeness_tailfeat_0.1\_NoUserloss_dev"
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        multiple_files=True \
#        file_extracted=False \
#        ###
#done
#
#TITLE_VERSION=seed10_gtp_informativeness_80_20_100
#TITLE_PATH="../results_important/s2_shared_own_userize/bart_input_pred_A_extra_online_A_userize_average_mum_loss_v4/checkpoint-8500/t80_v20_t100_seed10/userize_informativeness_tailfeat_0.1\_NoUserloss_dev"
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        multiple_files=True \
#        file_extracted=False \
#        ###
#done

#TITLE_VERSION=seed0_interuser_50_3_50_gtp
#TITLE_PATH=../results_important/s2_shared_own_userize/INTERUSER/50_3_50/bart_input_pred_A_extra_online_A_userize_average_mum_loss_v4/checkpoint-8500/gtp_run4
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        interuser=True \
#        ###
#done

#TITLE_VERSION=seed0_interuser_50_3_50_gtp
#TITLE_PATH=../results_important/s2_shared_own_userize/INTERUSER/50_3_50/bart_input_pred_A_extra_online_A_userize_average_mum_loss_v4/checkpoint-8500/gtp_run6
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        interuser=True \
#        ###
#done

#TITLE_VERSION=seed1_interuser_50_3_50_gtp
#TITLE_PATH=../results_important/s2_shared_own_userize/INTERUSER/50_3_50/bart_input_pred_A_extra_online_A_userize_average_mum_loss_v4/checkpoint-8500/seed1_gtp_run6
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        interuser=True \
#        ###
#done
#
#TITLE_VERSION=seed10_interuser_50_3_50_gtp
#TITLE_PATH=../results_important/s2_shared_own_userize/INTERUSER/50_3_50/bart_input_pred_A_extra_online_A_userize_average_mum_loss_v4/checkpoint-8500/seed10_gtp_run6
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        interuser=True \
#        ###
#done

#TITLE_VERSION=seed0_interuser_50_3_50_two_stage_userize
#TITLE_PATH=../results_important/s2_shared_own_userize/INTERUSER/50_3_50/bart-base/two_stage
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        interuser=True \
#        ###
#done
#
#TITLE_VERSION=seed1_interuser_50_3_50_two_stage_userize
#TITLE_PATH=../results_important/s2_shared_own_userize/INTERUSER/50_3_50/bart-base/seed1_two_stage_userize
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        interuser=True \
#        ###
#done
#
#TITLE_VERSION=seed10_interuser_50_3_50_two_stage_userize
#TITLE_PATH=../results_important/s2_shared_own_userize/INTERUSER/50_3_50/bart-base/seed10_two_stage_userize
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        interuser=True \
#        ###
#done

#TITLE_VERSION=seed0_interuser_50_3_50_two_stage_wo_userize
#TITLE_PATH=../results_important/s2_shared_own_userize/INTERUSER/50_3_50/bart-base/two_stage_wo_userize
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        interuser=True \
#        ###
#done

#TITLE_VERSION=seed1_interuser_50_3_50_two_stage_wo_userize
#TITLE_PATH=../results_important/s2_shared_own_userize/INTERUSER/50_3_50/bart-base/seed1_two_stage_wo_userize
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        interuser=True \
#        ###
#done
#
#TITLE_VERSION=seed10_interuser_50_3_50_two_stage_wo_userize
#TITLE_PATH=../results_important/s2_shared_own_userize/INTERUSER/50_3_50/bart-base/seed10_two_stage_wo_userize
#for ((FOLD_ID=0; FOLD_ID<=1; FOLD_ID++));
#do
#    CKPT=$(find ../results/ape/train/$MODEL_TYPE/all/fold_$FOLD_ID/version_0/checkpoints -name *.ckpt | head -n 1)
#    python main_ape.py with \
#        $MODEL_TYPE exp_name=$MODEL_TYPE/$TITLE_VERSION \
#        fold_id=$FOLD_ID fold_size=$FOLD_SIZE \
#        user_id=all checkpoint_path=$CKPT \
#        predict_title_file=$TITLE_PATH \
#        interuser=True \
#        ###
#done

####### Check APE ########

#BASE_VERSION=random200_editor_title
#python ape/result_fad.py \
#    --user_title_dir ../results/ape/predict/$MODEL_TYPE/user_title \
#    --base_title_dir ../results/ape/predict/$MODEL_TYPE/$BASE_VERSION \
#    --average_over \

####### Verify a specific model ######
#TITLE_VERSION=stage1
#BASE_VERSION=random200_editor_title
#python ape/result_fad.py \
#    --user_title_dir ../results/ape/predict/$MODEL_TYPE/user_title \
#    --base_title_dir ../results/ape/predict/$MODEL_TYPE/$BASE_VERSION \
#    --title_dir ../results/ape/predict/$MODEL_TYPE/$TITLE_VERSION \


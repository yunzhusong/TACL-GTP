"""
This file is to conduct the significace test for ROUGE scores and APE scores.
"""
import argparse
import os
import json
import random
import pandas as pd
from glob import glob
import numpy as np
from datasets import load_metric
from tqdm import tqdm
from scipy import linalg
import time

result_files = {
    #"chatGPT": "../results_important/chatGPT",
    #"stage1": "../results/ghg/bart/checkpoint-32000",
    #"stage1_old": "../results_important/pens_ghg_own/bart/checkpoint-98000/phg",
    #"early_fusion": "../results_important/s2_shared_own/bart_userize_average_v4/checkpoint-13000",
    ##"gtp_wo_penssh_old": "../results_important/specialize_own/bart_input_pred_A_extra_online_A_userize_mum_loss/checkpoint-2600",
    #"new_gtp": "../results/phg/TrRM_fixEmbed/checkpoint-84000/userize_10token_isb_rec_mum_2e-6lr_run2/checkpoint-12500",
    #"seed0_gtp_80_20_100": "../results_important/s2_shared_own_userize/bart_input_pred_A_extra_online_A_userize_average_mum_loss_v4/checkpoint-8500/userize_seed0_t80_v20_t100_average_NoUserloss_dev",
    #"seed1_gtp_80_20_100": "../results_important/s2_shared_own_userize/bart_input_pred_A_extra_online_A_userize_average_mum_loss_v4/checkpoint-8500/t80_v20_t100_seed1/userize_random_average_NoUserloss_dev",
    #"seed10_gtp_80_20_100": "../results_important/s2_shared_own_userize/bart_input_pred_A_extra_online_A_userize_average_mum_loss_v4/checkpoint-8500/t80_v20_t100_seed10/userize_random_average_NoUserloss_dev",
    #"seed0_hg_hc_80_20_100": "../results_important/s2_shared_own_userize/bart-base/no_extra_seed0_t80_v20_t100_dev",
    #"seed1_hg_hc_80_20_100": "../results_important/s2_shared_own_userize/bart-base/t80_v20_t100_seed1/no_extra_random_dev",
    #"seed10_hg_hc_80_20_100": "../results_important/s2_shared_own_userize/bart-base/t80_v20_t100_seed10/no_extra_random_dev",
    #"seed0_wo_TrRMIo_80_20_100": "../results_important/s2_shared_own_userize/bart_news_bart_input_pred_A_extra_online_A_userize_average_mum_loss_v4/checkpoint-8000/t80_v20_t100/userize_informativeness_tailfeat_275-NoUserloss_bart_news_dev",
    #"seed1_wo_TrRMIo_80_20_100": "../results_important/s2_shared_own_userize/bart_news_bart_input_pred_A_extra_online_A_userize_average_mum_loss_v4/checkpoint-8000/t80_v20_t100_seed1/userize_random_average_NoUserloss_dev",
    #"seed10_wo_TrRMIo_80_20_100": "../results_important/s2_shared_own_userize/bart_news_bart_input_pred_A_extra_online_A_userize_average_mum_loss_v4/checkpoint-8000/t80_v20_t100_seed10/userize_random_average_NoUserloss_dev",
    #"seed0_wo_mum_80_20_100": "../results_important/s2_shared_own_userize/bart_input_pred_A_extra_online_A_userize_average_v4/checkpoint-8000/t80_v20_t100/userize_informativeness_average_NoUserloss_dev",
    #"seed1_wo_mum_80_20_100": "../results_important/s2_shared_own_userize/bart_input_pred_A_extra_online_A_userize_average_v4/checkpoint-8000/t80_v20_t100_seed1/userize_random_average_NoUserloss_dev",
    #"seed10_wo_mum_80_20_100": "../results_important/s2_shared_own_userize/bart_input_pred_A_extra_online_A_userize_average_v4/checkpoint-8000/t80_v20_t100_seed10/userize_random_average_NoUserloss_dev",
    #"seed0_wo_isb_80_20_100": "../results_important/s2_shared_own_userize/bart_input_pred_A_userize_mum_loss/checkpoint-9500/t80_v20_t100/userize_no_extra_informativeness_average_NoUserloss_dev",
    #"seed1_wo_isb_80_20_100": "../results_important/s2_shared_own_userize/bart_input_pred_A_userize_mum_loss/checkpoint-9500/t80_v20_t100_seed1/userize_no_extra_random_average_NoUserloss_dev",
    #"seed10_wo_isb_80_20_100": "../results_important/s2_shared_own_userize/bart_input_pred_A_userize_mum_loss/checkpoint-9500/t80_v20_t100_seed10/userize_no_extra_random_average_NoUserloss_dev",
    #"seed0_lf_80_20_100": "../results_important/s2_shared_own_userize/bart-base/userize_no_extra_seed0_t80_v20_t100_dev",
    #"seed1_lf_80_20_100": "../results_important/s2_shared_own_userize/bart-base/t80_v20_t100_seed1/userize_no_extra_random_average_NoUserloss_dev",
    #"seed10_lf_80_20_100": "../results_important/s2_shared_own_userize/bart-base/t80_v20_t100_seed10/userize_no_extra_random_average_NoUserloss_dev",

    #"seed0_wo_lf_80_20_100": "../results_important/s2_shared_own_userize/bart_userize_average_v4/checkpoint-13000/userize_no_extra_seed0_t80_v20_t100_average_NoUserloss_dev",
    #"seed1_wo_lf_80_20_100": "../results_important/s2_shared_own_userize/bart_userize_average_v4/checkpoint-13000/userize_no_extra_seed1_t80_v20_t100_seed1_average_NoUserloss_dev",
    #"seed10_wo_lf_80_20_100": "../results_important/s2_shared_own_userize/bart_userize_average_v4/checkpoint-13000/userize_no_extra_seed10_t80_v20_t100_seed10_average_NoUserloss_dev",

    "seed0_interuser_50_3_50_gtp": "../results_important/s2_shared_own_userize/INTERUSER/50_3_50/bart_input_pred_A_extra_online_A_userize_average_mum_loss_v4/checkpoint-8500/gtp_run6",
    "seed1_interuser_50_3_50_gtp": "../results_important/s2_shared_own_userize/INTERUSER/50_3_50/bart_input_pred_A_extra_online_A_userize_average_mum_loss_v4/checkpoint-8500/seed1_gtp_run6",
    "seed10_interuser_50_3_50_gtp": "../results_important/s2_shared_own_userize/INTERUSER/50_3_50/bart_input_pred_A_extra_online_A_userize_average_mum_loss_v4/checkpoint-8500/seed10_gtp_run6",
    "seed0_interuser_50_3_50_two_stage_w_userize": "../results_important/s2_shared_own_userize/INTERUSER/50_3_50/bart-base/two_stage",
    "seed1_interuser_50_3_50_two_stage_w_userize": "../results_important/s2_shared_own_userize/INTERUSER/50_3_50/bart-base/seed1_two_stage_userize",
    "seed10_interuser_50_3_50_two_stage_w_userize": "../results_important/s2_shared_own_userize/INTERUSER/50_3_50/bart-base/seed10_two_stage_userize",
    "seed0_interuser_50_3_50_two_stage_wo_userize": "../results_important/s2_shared_own_userize/INTERUSER/50_3_50/bart-base/two_stage_wo_userize",
    "seed1_interuser_50_3_50_two_stage_wo_userize": "../results_important/s2_shared_own_userize/INTERUSER/50_3_50/bart-base/seed1_two_stage_wo_userize",
    "seed10_interuser_50_3_50_two_stage_wo_userize": "../results_important/s2_shared_own_userize/INTERUSER/50_3_50/bart-base/seed10_two_stage_wo_userize",
}

def get_result_file(key):
    if "80_20_100" in key:
        fewshot_setting = "userwise"
    elif "interuser" in key:
        fewshot_setting = "interuser"
    else:
        fewshot_setting = False

    extracted = "results_important" not in result_files[key]

    return result_files[key], fewshot_setting, extracted


def find_better_checkpoint(data_path):
    """ Return checkpoint with better performance """
    max_score = 0
    print(data_path)
    for checkpoint in glob(f"{data_path}/checkpoint*"):
        result_path = os.path.join(checkpoint, "all_results.json")
        #result_path = os.path.join(data_path, checkpoint, "all_results.json")

        f = open(result_path)
        results = json.load(f)
        score = results["predict_combined_score"]
        if score > max_score:
            best_checkpoint = checkpoint

    return best_checkpoint


def output(all_texts, is_test_data=None, output_path=None):
    os.makedirs(output_path, exist_ok=True)
    if is_test_data is not None:
        titles = [""] * len(is_test_data)
        cnt = 0
        for i, test_data in enumerate(is_test_data):
            if test_data:
                titles[i] = all_texts[cnt]
                cnt += 1
        assert (cnt == len(all_texts))

        with open(os.path.join(output_path, "generated_predictions.txt"), "w") as f:
            f.write("\n".join(titles))
    else:
        with open(os.path.join(output_path, "generated_predictions.txt"), "w") as f:
            f.write("\n".join(all_texts))



def collect_predictions(data_file, extracted, interuser=False):
    if interuser:
        """ Collect the prediction """
        all_texts = []
        better_checkpoint = find_better_checkpoint(data_file)
        with open(f"{better_checkpoint}/generated_predictions.txt", "r") as f:
            all_texts += [s.strip() for s in f.readlines()]
        all_splits = (pd.read_csv(data_file+"/split.csv")["split"] == "test").tolist()
        is_test_data = np.array(all_splits)

    elif extracted:
        """ Collect the prediction from users """
        all_texts = []
        for i in range(103):
            with open(f"{data_file}/NT{i+1}/generated_predictions.txt", "r") as f:
                texts = [s.strip() for s in f.read().split(";;")]
                all_texts += texts
        all_splits = (pd.read_csv(data_file+"/split.csv")["split"] == "test").tolist()
        is_test_data = np.array(all_splits*103)=="test"

    else:
        """ Need to find the better checkpoint """
        all_splits = []
        all_texts = []
        for i in range(103):
            better_checkpoint = find_better_checkpoint(f"{data_file}/index{i}")
            with open(f"{better_checkpoint}/generated_predictions.txt", "r") as f:
                all_texts += [s.strip() for s in f.readlines()]
            all_splits += pd.read_csv(f"{data_file}/index{i}/split.csv")["split"].tolist()
        is_test_data = np.array(all_splits)=="test"
    
    return all_texts, is_test_data


def get_rouge_scores():

    test_file = "../datasets/pens/test.pkl"
    # Read testing data and get ground-truth
    test_df = pd.read_pickle(test_file)
    test_titles = np.array(sum([i.split(";;") for i in test_df["title"].tolist()], []))

    # Load metric
    metric = load_metric("rouge")

    # List of result files
    for k in tqdm(result_files.keys()):

        #result_file, few_shot, extracted = get_result_file("seed0_gtp_80_20_100")
        result_file, few_shot, extracted = get_result_file(k)

        # Colecting prediction file if needed
        if few_shot=="userwise":
            predictions, is_test_data = collect_predictions(result_file, extracted)
            references = test_titles[is_test_data]
            output(predictions, is_test_data, os.path.join("/nfs/home/yunzhu/Workspace/PHG/results/final", k))
        elif few_shot=="interuser":
            predictions, is_test_data = collect_predictions(result_file, extracted=True, interuser=True)
            references = test_titles[is_test_data]
            output(predictions, is_test_data, os.path.join("/nfs/home/yunzhu/Workspace/PHG/results/final", k))
        else:
            with open(os.path.join(result_file, "generated_predictions.txt")) as f:
                predictions = [s.strip() for s in f.readlines()]

            references = test_titles
            output(predictions, is_test_data=None, output_path=os.path.join("/nfs/home/yunzhu/Workspace/PHG/results/final", k))

        # Slice the ground-truth by spliting

        # Get ROUGE scores without aggregations
        result = metric.compute(predictions=predictions, references=references,
                                use_stemmer=True, use_aggregator=False)

        print(k)
        print(result)

        # Write out the scores
        np.savez(f"{result_file}/rouge_scores.npz",
                rouge1=np.array([s.fmeasure for s in result["rouge1"]]),
                rouge2=np.array([s.fmeasure for s in result["rouge2"]]),
                rougeL=np.array([s.fmeasure for s in result["rougeL"]]),
                rougeLsum=np.array([s.fmeasure for s in result["rougeLsum"]]),
        )

def bootstrapping(model1, model2, res_model1=None, res_model2=None):

    # H0: model 1 and model 2 are the same (mean difference == 0)
    # H1: model 1 is greater than model 2 (mean difference > 0)

    # Load the ROUGE scores of model1 and model2
    if res_model1 is None:
        res_model1 = np.load(f"{result_files[model1]}/rouge_scores.npz")
        res_model2 = np.load(f"{result_files[model2]}/rouge_scores.npz")

    random.seed(0)
    n_resamples = 10000
    sample_size = 10000

    print("****")
    print(f"H1: {model1} is greater than {model2}")
    # bootstrapping using a subset of data
    p_values = []
    rouges_model1 = []
    rouges_model2 = []
    for metric in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
        res = res_model1[metric] - res_model2[metric]
        diff = np.mean(res_model1[metric]) - np.mean(res_model2[metric])
        score_model1 = np.mean(res_model1[metric])
        score_model2 = np.mean(res_model2[metric])
        diff = score_model1 - score_model2
        #print(f"{score_model1:.4f} -  {score_model2:.4f} = {diff:.4f}")
        rouges_model1.append(score_model1)
        rouges_model2.append(score_model2)

        means = []
        for _ in range(n_resamples):
            # sample
            random_index = np.random.randint(0, len(res), sample_size)
            means.append(np.mean(res[random_index]))

        null_val = np.random.normal(0, np.std(means), sample_size)
        p_value = (null_val>diff).mean()
        #print(f"{metric}'s p_value: {p_value:.6f}")
        p_values.append(p_value)
    print(rouges_model1)
    print(rouges_model2)
    print(p_values)

    return p_values, rouges_model1, rouges_model2


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
		    d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
	    mu1: Numpy array containing the activations of a layer of the
		    inception net (like returned by the function 'get_predictions')
		    for generated samples.
	    mu2: The sample mean over activations, precalculated on an
		    representative data set.
	    sigma1: The covariance matrix over activations for generated samples.
	    sigma2: The covariance matrix over activations, precalculated on an
		    representative data set.
    Returns:
	    The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
           'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print('fid calculation produces singular product; '
             f'adding {eps} to diagonal of cov estimates')
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    return (diff.dot(diff) + 
            np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))

def collect_statistics(title_dir):
    # Collect score and mask files
    features = []
    care_data_mask = []

    user_dirs = os.listdir(title_dir)
    user_dirs = sorted(user_dirs)
    for ud in user_dirs:
        fold_dirs = os.listdir(f"{title_dir}/{ud}")
        fold_dirs = sorted(fold_dirs)
        for fd in fold_dirs:

            statistics = np.load(f"{title_dir}/{ud}/{fd}/statistic_bottleneck.npz")
            features.append(statistics["features"])
            care_data_mask.append(statistics["eval_care_data_mask"])

    return features, care_data_mask

def get_ape_score(user_feat, gen1_feat, base_feat):
    mu_user, sigma_user = np.mean(user_feat), np.cov(user_feat, rowvar=False)
    mu_gen1, sigma_gen1 = np.mean(gen1_feat), np.cov(gen1_feat, rowvar=False)
    mu_base, sigma_base = np.mean(base_feat), np.cov(base_feat, rowvar=False)

    d_base = calculate_frechet_distance(mu_user, sigma_user, mu_base, sigma_base)
    d_gen1 = calculate_frechet_distance(mu_user, sigma_user, mu_gen1, sigma_gen1)

    s_gen1 = d_gen1 / d_base

    return s_gen1


def get_ape_scores(user_feat, gen1_feat, gen2_feat, base_feat):
    mu_user, sigma_user = np.mean(user_feat), np.cov(user_feat, rowvar=False)
    mu_gen1, sigma_gen1 = np.mean(gen1_feat), np.cov(gen1_feat, rowvar=False)
    mu_gen2, sigma_gen2 = np.mean(gen2_feat), np.cov(gen2_feat, rowvar=False)
    mu_base, sigma_base = np.mean(base_feat), np.cov(base_feat, rowvar=False)

    d_base = calculate_frechet_distance(mu_user, sigma_user, mu_base, sigma_base)
    d_gen1 = calculate_frechet_distance(mu_user, sigma_user, mu_gen1, sigma_gen1)
    d_gen2 = calculate_frechet_distance(mu_user, sigma_user, mu_gen2, sigma_gen2)

    s_gen1 = d_gen1 / d_base
    s_gen2 = d_gen2 / d_base

    diff = s_gen2 - s_gen1

    return diff, s_gen1, s_gen2

def bootstrapping_ape(model1, model2):
    
    # Get title directory and mask directory
    base_model = "random200_editor_title"
    user_model = "user_title"
    root_dir = "../results/ape/predict/transformer_bottleneck"

    # Load features for both models
    user_features, user_masks = collect_statistics(os.path.join(root_dir, user_model))
    base_features, base_masks = collect_statistics(os.path.join(root_dir, base_model))
    gen1_features, gen1_masks = collect_statistics(os.path.join(root_dir, model1))
    gen2_features, gen2_masks = collect_statistics(os.path.join(root_dir, model2))
    
    #assert (gen1_masks[0] == gen2_masks[0]).any()
    cares = [True if sum(mask)>0 else False for mask in gen1_masks]
    masks = gen1_masks

    n_resamples = 100
    sample_size = 1000

    # Calculate time
    start_time = time.time()

    p_values = []
    ape_model1 = []
    ape_model2 = []
    print("*****")
    print(f"H1: {model1} is better than {model2}")
    for i in range(2):
        if not cares[i]:
            continue

        # Leave the care data
        user_feat = user_features[i][masks[i]]
        gen1_feat = gen1_features[i][masks[i]]
        gen2_feat = gen2_features[i][masks[i]]
        base_feat = base_features[i][masks[i]]

        # Get the APE scores and the difference between two models
        diff, s_gen1, s_gen2 = get_ape_scores(user_feat, gen1_feat, gen2_feat, base_feat)
        ape_model1.append(s_gen1)
        ape_model2.append(s_gen2)

        # Resampling for both models
        all_s_gen1, all_s_gen2 = [], []
        for _ in range(n_resamples):
            index = np.random.randint(0, len(user_feat), sample_size)
            # Get frecht distance for both models
            s_gen1 = get_ape_score(
                user_feat[index], gen1_feat[index], base_feat[index]
            )
            all_s_gen1.append(s_gen1)

        # Build normal distribution with zero mean
        null_val = np.random.normal(0, np.std(all_s_gen1), 10000)
        p_value = (null_val>diff).mean()
        p_values.append(p_value)

        print(f"APEs: {s_gen1:.4f}, {s_gen2:.4f}")
        print(f"Std of {model1}: {np.std(all_s_gen1):.4f}")

    if np.mean(p_values) < 0.05:
        print(f"Significant !!")
    else:
        print(f"Not significant.")
    print(f"Difference: {diff:.4f} ")
    print(f"p_values: {p_values} -> Average: {np.mean(p_values):.4f}")
    print("Take {} mins to compute".format((time.time()-start_time)//60))

    return np.mean(p_values), np.mean(ape_model1), np.mean(ape_model2)


def main_rouge_boot_system_input():
    """ Significance test for ROUGE scores. Boostrapping for both input samples and different experiment runs """

    from collections import defaultdict
    out_dir = "../results/significant_test"
    os.makedirs(out_dir, exist_ok=True)

    #### Few-shot ####
    df_p_values = defaultdict(list)
    df_rouges = defaultdict(list)


    model1 = f"gtp_80_20_100" # our
    model2_list = [
        f"hg_hc_80_20_100",
        f"wo_TrRMIo_80_20_100",
        f"wo_mum_80_20_100",
        f"wo_isb_80_20_100",
        f"lf_80_20_100",
        f"wo_lf_80_20_100",
    ]

    for model2 in model2_list:
        for s in [10, 1, 0]:
            seed = f"seed{s}"
        res_model1_seeds = [np.load(f"{result_files[s+'_'+model1]}/rouge_scores.npz") for s in ["seed10", "seed1", "seed0"]]
        res_model2_seeds = [np.load(f"{result_files[s+'_'+model2]}/rouge_scores.npz") for s in ["seed10", "seed1", "seed0"]]
        res_model1 = {}
        res_model2 = {}
        for m in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
            res_model1[m] = np.concatenate([res[m] for res in res_model1_seeds])
            res_model2[m] = np.concatenate([res[m] for res in res_model2_seeds])

        p_values, rouges_model1, rouges_model2 = bootstrapping(
            f"{model1}", f"{model2}", res_model1, res_model2
        )
    
        df_p_values[model2].append(p_values)
        df_rouges[model2].append(rouges_model2)
    df_rouges[model1].append(rouges_model1)

    for k, v in df_p_values.items():
        df_p_values[k].append(np.array(v).mean(axis=0))

    for k, v in df_rouges.items():
        df_rouges[k].append(np.array(v).mean(axis=0))

    # Write out
    df_p_values = pd.DataFrame(df_p_values)
    df_p_values.to_csv(out_dir+"/rouges_pvalues_fewshot.csv")
    df_rouges = pd.DataFrame(df_rouges)
    df_rouges.to_csv(out_dir+"/rouges_fewshot.csv")

    #### Zero-shot ####
    df_p_values = defaultdict(list)
    df_rouges = defaultdict(list)

    model1 = "new_gtp"
    model2_list = [
        "chatGPT",
        "stage1",
        #"early_fusion",
        #"gtp_wo_penssh_old",
        #"stage1_old",
    ]

    for model2 in model2_list:
        p_values, rouges_model1, rouges_model2 = bootstrapping(model1, model2)

        df_p_values[model2].append(p_values)
        df_rouges[model2].append(rouges_model2)
    df_rouges[model1].append(rouges_model1)


    df_rouges = pd.DataFrame(df_rouges)
    df_p_values = pd.DataFrame(df_p_values)
    df_p_values.to_csv(out_dir+"/rouges_pvalues_zeroshot.csv")
    df_rouges.to_csv(out_dir+"/rouges_zeroshot.csv")

def main_rouge_boot_input():
    """ Significant test for ROUGE scores. Only bootstrapping the input sample. """

    from collections import defaultdict
    out_dir = "../results/significant_test"
    os.makedirs(out_dir, exist_ok=True)

    #### Few-shot ####
    df_p_values = defaultdict(list)
    df_rouges = defaultdict(list)

    for s in [10, 1, 0]:

        model1 = f"gtp_80_20_100" # our
        model2_list = [
            f"hg_hc_80_20_100",
            f"wo_TrRMIo_80_20_100",
            f"wo_mum_80_20_100",
            f"wo_isb_80_20_100",
            f"lf_80_20_100",
            f"wo_lf_80_20_100",
        ]

        seed = f"seed{s}"
        for model2 in model2_list:
            p_values, rouges_model1, rouges_model2 = bootstrapping(
                f"{seed}_{model1}", f"{seed}_{model2}")
    
            df_p_values[model2].append(p_values)
            df_rouges[model2].append(rouges_model2)
        df_rouges[model1].append(rouges_model1)

    for k, v in df_p_values.items():
        df_p_values[k].append(np.array(v).mean(axis=0))

    for k, v in df_rouges.items():
        df_rouges[k].append(np.array(v).mean(axis=0))

    # Write out
    df_p_values = pd.DataFrame(df_p_values)
    df_p_values.to_csv(out_dir+"/rouges_pvalues_fewshot.csv")
    df_rouges = pd.DataFrame(df_rouges)
    df_rouges.to_csv(out_dir+"/rouges_fewshot.csv")


    #### Zero-shot ####
    df_p_values = defaultdict(list)
    df_rouges = defaultdict(list)

    model1 = "new_gtp"
    model2_list = [
        "stage1",
        #"early_fusion",
        #"gtp_wo_penssh_old",
        #"stage1_old",
    ]

    for model2 in model2_list:
        p_values, rouges_model1, rouges_model2 = bootstrapping(model1, model2)

        df_p_values[model2].append(p_values)
        df_rouges[model2].append(rouges_model2)
    df_rouges[model1].append(rouges_model1)


    df_rouges = pd.DataFrame(df_rouges)
    df_p_values = pd.DataFrame(df_p_values)
    df_p_values.to_csv(out_dir+"/rouges_pvalues_zeroshot.csv")
    df_rouges.to_csv(out_dir+"/rouges_zeroshot.csv")

def main_rouge_boot_input_interuser():
    """ Significant test for ROUGE scores. Only bootstrapping the input sample. """

    from collections import defaultdict
    out_dir = "../results/significant_test_interuser"
    os.makedirs(out_dir, exist_ok=True)

    #### Few-shot ####
    df_p_values = defaultdict(list)
    df_rouges = defaultdict(list)

    for s in [10, 1, 0]:

        model1 = "interuser_50_3_50_gtp" # our
        model2_list = [
            "interuser_50_3_50_two_stage_wo_userize"
        ]

        seed = f"seed{s}"
        for model2 in model2_list:
            p_values, rouges_model1, rouges_model2 = bootstrapping(
                f"{seed}_{model1}", f"{seed}_{model2}")
    
            df_p_values[model2].append(p_values)
            df_rouges[model2].append(rouges_model2)
        df_rouges[model1].append(rouges_model1)

    for k, v in df_p_values.items():
        df_p_values[k].append(np.array(v).mean(axis=0))

    for k, v in df_rouges.items():
        df_rouges[k].append(np.array(v).mean(axis=0))

    # Write out
    df_p_values = pd.DataFrame(df_p_values)
    df_p_values.to_csv(out_dir+"/rouges_pvalues_fewshot.csv")
    df_rouges = pd.DataFrame(df_rouges)
    df_rouges.to_csv(out_dir+"/rouges_fewshot.csv")


def main_ape_boot_interuser():
    """ Significance test for APE scores. Bootstrapping for both input samples and different experimental runs. """

    from collections import defaultdict
    out_dir = "../results/significant_test_interuser"
    os.makedirs(out_dir, exist_ok=True)

    #### Few-shot ####
    # Prepare to collect
    df_p_values = defaultdict(list)
    df_ape = defaultdict(list)

    # Different seeds
    for s in [0]:
        model1 = "interuser_50_3_50_gtp" # our
        model2_list = [
            "interuser_50_3_50_two_stage_wo_userize"
        ]

        seed = f"seed{s}"

        for model2 in model2_list:
            p_values, ape_model1, ape_model2 = bootstrapping_ape(
                f"{seed}_{model1}", f"{seed}_{model2}")
            df_p_values[f"{model2}"].append(p_values)
            df_ape[f"{model2}"].append(ape_model2)
        df_ape[f"{model1}"].append(ape_model1)

    # Average over seeds
    for k, v in df_p_values.items():
        df_p_values[k].append(np.array(v).mean(axis=0))

    for k, v in df_ape.items():
        df_ape[k].append(np.array(v).mean(axis=0))

    # Write out
    df_p_values = pd.DataFrame(df_p_values)
    df_p_values.sort_index(axis=1, inplace=True)
    df_p_values.to_csv(f"{out_dir}/ape_pvalues_fewshot.csv")
    
def main_ape_boot():
    """ Significance test for APE scores. Bootstrapping for both input samples and different experimental runs. """

    from collections import defaultdict
    out_dir = "../results/significant_test"
    os.makedirs(out_dir, exist_ok=True)

    #### Few-shot ####
    # Prepare to collect
    df_p_values = defaultdict(list)
    df_ape = defaultdict(list)

    # Different seeds
    for s in [10, 1, 0]:
        model1 = f"gtp_80_20_100" # our
        model2_list = [
            f"hg_hc_80_20_100",
            f"wo_TrRMIo_80_20_100",
            f"wo_mum_80_20_100",
            f"wo_isb_80_20_100",
            f"lf_80_20_100",
            f"wo_lf_80_20_100",
        ]
        seed = f"seed{s}"

        for model2 in model2_list:
            p_values, ape_model1, ape_model2 = bootstrapping_ape(
                f"{seed}_{model1}", f"{seed}_{model2}")
            df_p_values[f"{model2}"].append(p_values)
            df_ape[f"{model2}"].append(ape_model2)
        df_ape[f"{model1}"].append(ape_model1)

    # Average over seeds
    for k, v in df_p_values.items():
        df_p_values[k].append(np.array(v).mean(axis=0))

    for k, v in df_ape.items():
        df_ape[k].append(np.array(v).mean(axis=0))

    # Write out
    df_p_values = pd.DataFrame(df_p_values)
    df_p_values.sort_index(axis=1, inplace=True)
    df_p_values.to_csv(f"{out_dir}/ape_pvalues_fewshot.csv")
    
    df_ape = pd.DataFrame(df_ape)
    df_ape.sort_index(axis=1, inplace=True)
    df_ape.to_csv(f"{out_dir}/ape_fewshot.csv")

    #### Zero-shot ####
    # Prepare to collect
    df_p_values = defaultdict(list)
    df_ape = defaultdict(list)

    model1 = "new_gtp"
    model2_list = [
        "chatGPT",
        "stage1",
        #"stage1_old",
        "early_fusion",
        "gtp_wo_penssh_old",
    ]

    for model2 in model2_list:
        p_values, ape_model1, ape_model2 = bootstrapping_ape(model1, model2)
        df_p_values[f"{model1}_vs_{model2}"].append(p_values)
        df_ape[f"{model2}"].append(ape_model2)
    df_ape[f"{model1}"].append(ape_model1)

    df_p_values = pd.DataFrame(df_p_values)
    df_p_values.sort_index(axis=1, inplace=True)
    df_p_values.to_csv(f"{out_dir}/ape_pvalues_zeroshot.csv")
    
    df_ape = pd.DataFrame(df_ape)
    df_ape.sort_index(axis=1, inplace=True)
    df_ape.to_csv(f"{out_dir}/ape_zeroshot.csv")


if __name__=="__main__":


    ### Significace test for ROUGE ###
    get_rouge_scores()
     
    #main_rouge_boot_input() # Bootstrapping for only input sample
    main_rouge_boot_input_interuser() # Bootstrapping for only input sample
    #main_rouge_boot_system_input() # Bootstapping for both input and systems

    ### Significace test for APE
    #main_ape_boot()

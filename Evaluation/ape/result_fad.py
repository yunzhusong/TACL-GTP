import os
import argparse

import numpy as np
from scipy import linalg
from collections import defaultdict

import pandas as pd

# NOTE: This is for the results of user study.
np.random.seed(200)
random_index = np.arange(10300)
np.random.shuffle(random_index)

def collect_statistics(title_dir, max_user):
    # Collect score and mask files
    mu = []
    sigma = []
    features = []
    care_data_mask = []

    dir_name = title_dir.split("/")[-1]
    user_dirs = os.listdir(title_dir)
    user_dirs = sorted(user_dirs)
    if max_user is not None:
        user_dirs = [i for i in user_dirs if int(i[2:]) <= max_user]
    for ud in user_dirs:
        fold_dirs = os.listdir(f"{title_dir}/{ud}")
        fold_dirs = sorted(fold_dirs)
        for fd in fold_dirs:

            statistics = np.load(f"{title_dir}/{ud}/{fd}/statistic_bottleneck.npz")
            mu.append(statistics["mu"])
            sigma.append(statistics["sigma"])
            features.append(statistics["features"])
            care_data_mask.append(statistics["eval_care_data_mask"])

    return mu, sigma, features, care_data_mask

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


#def main(args, verbose=True, base_reduction=0):
def main(title_dir, user_title_dir, mask_dir, base_title_dir, max_user, verbose=True, base_reduction=0):

    #dir_name = args.title_dir.split("/")[-1]
    #mask_name = args.title_dir.split("/")[-1]
    dir_name = title_dir.split("/")[-1]
    mask_name = title_dir.split("/")[-1]
    
    ref_mu, ref_sigma, ref_features, ref_masks = collect_statistics(user_title_dir, max_user)
    gen_mu, gen_sigma, gen_features, gen_masks = collect_statistics(title_dir, max_user)

    # Mask the samples used as the training data (generation models or ape models)
    masks = gen_masks
    if mask_dir is not None:
        mask_name = mask_dir.split("/")[-1]
        _, _, _, masks = collect_statistics(mask_dir, max_user)

    cares = [True if sum(mask)>0 else False for mask in masks]

    scores = []
    for ref_feat, gen_feat, mask, care in zip(ref_features, gen_features, masks, cares):
        if not care:
            continue

        ref_feat = ref_feat[mask]
        gen_feat = gen_feat[mask]

        mu1, sigma1 = np.mean(ref_feat), np.cov(ref_feat, rowvar=False)
        mu2, sigma2 = np.mean(gen_feat), np.cov(gen_feat, rowvar=False)

        scores.append(calculate_frechet_distance(mu1, sigma1, mu2, sigma2))

    if base_title_dir is not None:
        base_mu, base_sigma, base_features, _ = collect_statistics(base_title_dir, max_user)

        base_scores = []
        for ref_feat, base_feat, mask, care in zip(ref_features, base_features, masks, cares):
            if not care:
                continue

            ref_feat = ref_feat[mask]
            if base_reduction > 0:
                # For user study, where the sample number is very small
                base_feat = base_feat[random_index[:sum(mask)-base_reduction]]
            else:
                base_feat = base_feat[mask]
            
 
            mu1, sigma1 = np.mean(ref_feat), np.cov(ref_feat, rowvar=False)
            mu2, sigma2 = np.mean(base_feat), np.cov(base_feat, rowvar=False)

            base_scores.append(calculate_frechet_distance(mu1, sigma1, mu2, sigma2))

        ratio = []
        for s, b in zip(scores, base_scores):
            ratio.append(1-(s/b))
        ratio = np.mean(ratio)

        ratios = [(s/b) for s,b in zip(scores, base_scores)]
        ratio_mean = np.mean(ratios)
        ratio_var = np.var(ratios)
 
    if verbose:
        print(f"Score mean: {np.mean(scores):.4f} | Ratio var: {ratio_var:.4f} | Ratio: {ratio_mean:.4f}| Type: {dir_name} | Mask: {mask_name} | Base mean: {np.mean(base_scores):.4f}")
        print(scores)

    return {
        "mean": np.mean(scores),
        "var": ratio_var,
        "ratio": ratio_mean,
    }

def average_over_models(args, root_dir, title_version_list, mask_version_list):
    all_results = {}
    for title_version, title_dirs in title_version_list.items():
        all_results[title_version] = defaultdict(list)
            
        for i, title_dir in enumerate(title_dirs):
            #args.title_dir = os.path.join(root_dir, title_dir)
            mask_dir = mask_version_list[title_version][i]
            #args.mask_dir = os.path.join(root_dir, mask_dir)

            #result = main(args, verbose=False)
            result = main(
                title_dir=os.path.join(root_dir, title_dir),
                mask_dir=os.path.join(root_dir, mask_dir),
                user_title_dir=args.user_title_dir, 
                base_title_dir=args.base_title_dir,
                max_user=args.max_user,
                verbose=False,
            )
            for k, v in result.items():
                all_results[title_version][k].append(v)

            #args.mask_dir = None

        mean = np.mean(all_results[title_version]["mean"]) # mean of scores for different seeds
        var = np.var(all_results[title_version]["ratio"]) # variance of different seeds
        ratio = np.mean(all_results[title_version]["ratio"]) # mean of ratio for different seeds
        print(f"# of models: {len(title_dirs)} | mean: {mean:.4f} | var: {var:.4f} | ratio: {ratio:.4f} | Vesion: {title_version} | Mask: {mask_dir}")
        print(all_results[title_version]["ratio"])
        all_results[title_version] = {k: np.mean(v) for k, v in all_results[title_version].items()}
    return all_results


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--title_dir", type=str, default=None)
    parser.add_argument("--user_title_dir", type=str, default=None)
    parser.add_argument("--base_title_dir", type=str, default=None)
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument("--max_user", type=int, default=None)
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--average_over", action="store_true", default=False)
    parser.add_argument("--user_study", action="store_true", default=False)
    args = parser.parse_args()

    if args.title_dir is not None:
        main(args, verbose=True)
    
    elif args.title_dir is None and args.average_over:
        root_dir = "../results/ape/predict/transformer_bottleneck"
        title_version_list = {
            # Zero-shot
            "user_title": ["user_title"],
            "chatGPT": ["chatGPT"],
            "random200_editor_title": ["random200_editor_title"],
            "editor_title": ["editor_title"],
            "hg": ["stage1"],
            "early_fusion": ["early_fusion"],
            "gtp_wo_penssh": ["gtp_wo_penssh"],
            "gtp_wo_penssh_old": ["gtp_wo_penssh_old", "gtp_wo_penssh_1", "gtp_wo_penssh_2"],
            "new_gtp": ["new_gtp"],

            # Intrauser few-shot (some of them are not finetuned)
            "new_gtp_3runs": ["new_gtp", "new_gtp_run2", "new_gtp_run3"],

            "editor_title_3splits": ["editor_title" for i in [0,1,10]],
            "hg_3splits": [f"stage1" for i in [0,1,10]],
            "hg_hc_80_20_100": [f"seed{i}_hg_hc_80_20_100" for i in [0,1,10]],
            "gtp_80_20_100": [f"seed{i}_gtp_80_20_100" for i in [0,1,10]],
            "lf_80_20_100": [f"seed{i}_lf_80_20_100" for i in [0,1,10]],
            "wo_TrRMIo_80_20_100": [f"seed{i}_wo_TrRMIo_80_20_100" for i in [0,1,10]],
            "wo_mum_80_20_100": [f"seed{i}_wo_mum_80_20_100" for i in [0,1,10]],
            "wo_isb_80_20_100": [f"seed{i}_wo_isb_80_20_100" for i in [0,1,10]],
            "wo_lf_80_20_100": [f"seed{i}_wo_lf_80_20_100" for i in [0,1,10]],
            "wo_penssh_80_20_100": [f"seed{i}_gtp_wo_penssh_80_20_100" for i in [0]],
            "gtp_random_80_20_100": [f"seed{i}_gtp_random_80_20_100" for i in [0,1,10]],
            "gtp_diversity_80_20_100": [f"seed{i}_gtp_diversity_80_20_100" for i in [0,1,10]],
            "gtp_informativeness_80_20_100": [f"seed{i}_gtp_informativeness_80_20_100" for i in [0,1,10]],

            # Interuser few-shot (some of them are not finetuned)
            "interuser_one_stage": ["stage1" for i in [0,1,10]],
            "interuser_50_3_50_two_stage_userize": [f"seed{i}_interuser_50_3_50_two_stage_userize" for i in [0,1,10]],
            "interuser_50_3_50_two_stage_wo_userize": [f"seed{i}_interuser_50_3_50_two_stage_wo_userize" for i in [0,1,10]],
            "interuser_50_3_50_gtp": [f"seed{i}_interuser_50_3_50_gtp" for i in [0,1,10]],
        }
        # Assign the mask for the models that are not finetined
        mask_version_list = title_version_list.copy()
        mask_version_list["editor_title_3splits"] = [f"seed{i}_gtp_80_20_100" for i in [0,1,10]]
        mask_version_list["hg_3splits"] = [f"seed{i}_gtp_80_20_100" for i in [0,1,10]]
        mask_version_list["interuser_one_stage"] = [f"seed{i}_interuser_50_3_50_gtp" for i in [0,1,10]]

        all_results = average_over_models(args, root_dir, title_version_list, mask_version_list)
    
    if args.user_study:

        predict_dir = "../results/ape/predict/user_study"
        base_title_dir = "../results/ape/predict/transformer_bottleneck/random200_editor_title"
        user_title_dir = "../results/ape/predict/transformer_bottleneck/user_title"

        exp_results = dict()

        # Iterate over exp_name for user_study
        effective_exps = ["gtp-zs_vs_early", "gtp_vs_hg", "gtp_vs_editor"]
        for exp_name in effective_exps:
            if exp_name not in effective_exps:
                continue
    
            # Iterate over user_name
            ape_results = defaultdict(list)
            mean_results = defaultdict(list)
            positive_better = []
            user_names = os.listdir(os.path.join(predict_dir, exp_name))
            user_names = sorted(user_names)
            for user_name in user_names:

                # Iterate over positive and negative
                ratios = []
                means = []
                ape_results["user_name"].append(user_name)
                for result_type in ["positive", "negative"]:
                    title_dir = os.path.join(predict_dir, exp_name, user_name, result_type)
                    args.user_title_dir = user_title_dir
                    args.base_title_dir = base_title_dir
                    args.title_dir = title_dir
                    result = main(args, verbose=False, base_reduction=3)
                    ratio = result["ratio"]
                    mean = result["mean"]
                    ape_results[result_type].append(ratio)
                    mean_results[result_type].append(mean)
                    ratios.append(ratio)
                    means.append(mean)
                
                # Positive better
                if ratios[0] < ratios[1]:
                    positive_better.append(1)
                else:
                    positive_better.append(0)

            print(f"---- Experiment Name: {exp_name} ----")
            print(f"Percent of positive better: {np.average(positive_better):.4f}")
            print(f"# of user: {len(user_names)}")
            all_pos_ape = ape_results["positive"]
            all_neg_ape = ape_results["negative"]
            all_pos_mean = mean_results["positive"]
            all_neg_mean = mean_results["negative"]

            print(f"Mean: {np.average(all_pos_mean):.4f}, APE of positive: {np.average(all_pos_ape):.4f} ({np.var(all_pos_ape):.4f})")
            print(f"Mean: {np.average(all_neg_mean):.4f}, APE of negative: {np.average(all_neg_ape):.4f} ({np.var(all_neg_ape):.4f})")

            pd.DataFrame(ape_results).to_csv(f"../results/user_study/{exp_name}_ape.csv", index=False)

""" Program Process """
import os
import pdb
import logging
import numpy as np
import json
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)

def train_process(training_args, data_args, trainer, train_dataset):
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    # NEW:
    #ignore_keys_for_eval = ["rec_loss", "ctr_loss", "user_vec", "pred_ctr"]
    ignore_keys_for_eval = ["rec_loss", "ctr_loss", "user_vec", "pred_ctr"]
    train_result = trainer.train(resume_from_checkpoint=checkpoint, ignore_keys_for_eval=ignore_keys_for_eval)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


def eval_process(training_args, data_args, trainer, eval_dataset):
    #max_length = (
    #    training_args.generation_max_length
    #    if training_args.generation_max_length is not None
    #    else data_args.val_max_target_length
    #)
    ignore_keys = ["rec_loss", "ctr_loss", "user_vec", "pred_ctr"]
    #num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    metrics = trainer.evaluate(metric_key_prefix="eval", ignore_keys=ignore_keys)

    max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


def predict_process(training_args, data_args, trainer, predict_dataset, tokenizer):
    #max_length = (
    #    training_args.generation_max_length
    #    if training_args.generation_max_length is not None
    #    else data_args.val_max_target_length
    #)
    #num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams

    ignore_keys = ["rec_loss", "ctr_loss", "user_vec", "pred_ctr"]
    predict_results = trainer.predict(
        predict_dataset, metric_key_prefix="predict", ignore_keys=ignore_keys
    )
    metrics = predict_results.metrics
    max_predict_samples = (
        data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
    )
    metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)

def get_news_features(training_args, trainer, predict_dataset):
    news_features = trainer.news_features(predict_dataset)
    np.savez(training_args.output_dir+"/news_features.npz",
            news_features=news_features)

def get_recommended_score(training_args, trainer, predict_dataset):
    scores = trainer.predict_recommended_score(predict_dataset)
    #with open(training_args.output_dir+"/all_results.json") as f:
    #    results = json.load(f)
    #results["recommended_score"] = round(np.average(scores),4)

    np.savez(training_args.output_dir+"/recommended_scores.npz",
            recommended_scores=scores)

    #with open(training_args.output_dir+"/all_results.json", "w") as f:
    #    f.write(json.dumps(results, indent=4))


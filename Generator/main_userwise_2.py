"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import pdb
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric
import torch

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

# NEW: for copy script file
import shutil
# NEW: get arguments
from config.arguments import ModelArguments, DataTrainingArguments, CustomSeq2SeqTrainingArguments
# NEW:
from data.build_datasets_userwise import build_datasets
from pipelines.build_trainer import build_trainer

from process import train_process, eval_process, predict_process
from model.UTC import UTCBart
from utils.gadget import remove_files

from tqdm import tqdm
import pandas as pd
from data.utils import add_padding
from data.utils import load_user_features

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.12.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

def _find_best_checkpoint(model_path, score_type="max"):
    """ We define the best checkpoint as the later checkpoint with either lowest or highest rouge"""
    if score_type=="max":
        max_step = 0
        for filename in os.listdir(model_path):
            if "checkpoint" in filename:
                step = int(filename.replace("checkpoint-", ""))
                if step > max_step:
                    max_step = step
                    best_checkpoint = filename

        return os.path.join(model_path, best_checkpoint)

    if score_type=="min":
        min_step = 10000000
        for filename in os.listdir(model_path):
            if "checkpoint" in filename:
                step = int(filename.replace("checkpoint-", ""))
                if step < min_step:
                    min_step = step
                    best_checkpoint = filename

        return os.path.join(model_path, best_checkpoint)

def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomSeq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    #logging.basicConfig(
    #    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    #    datefmt="%m/%d/%Y %H:%M:%S",
    #    handlers=[logging.StreamHandler(sys.stdout)],
    #)
    #log_level = training_args.get_process_log_level()
    #log_level = logging.WARNING
    #logger.setLevel(log_level)
    #datasets.utils.logging.set_verbosity(log_level)
    #transformers.utils.logging.set_verbosity(log_level)
    #transformers.utils.logging.enable_default_handler()
    #transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    #logger.warning(
    #    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    #    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    #)
    #logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        #bart_local_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    )

    # Adjust model_input_names to fit the model
    config.userize = training_args.userize
    config.userize_loss = training_args.userize_loss
    config.userize_mum = training_args.userize_mum
    config.userize_dot = training_args.userize_dot
    config.userize_complex_proj = training_args.userize_complex_proj
    config.userize_type_embedding = training_args.userize_type_embedding
    config.userize_ufeat_type = training_args.userize_ufeat_type

    if training_args.userize:
        #if training_args.userize_loss:
        #    tokenizer.add_tokens(["<user>"])
        tokenizer.add_tokens(["<user>"])
        tokenizer.user_token = "<user>"
        tokenizer.user_token_id = tokenizer.vocab["<user>"]

        tokenizer.model_input_names.append("user_features")
        tokenizer.model_input_names.append("user_embeds")
        special_tokens_dict = {"additional_special_tokens": ["<user>"]}
        num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

        config.user_token_length = training_args.userize_user_token_length
        config.decoder_user_token_id = tokenizer.user_token_id


    # // Prepare userwise training
    news_file = "../datasets/pens/news.pkl"
    news = pd.read_pickle(news_file)
    news = news.fillna("")

    # Prepare inputs (if text_file is None, take ground truth headline as input to training)
    if training_args.text_file is not None:
        # title'
        raw_text = pd.read_csv(training_args.text_file, sep="\n", header=None)[0].tolist()
    else:
        # title
        text_column = data_args.text_column if data_args.text_column is not None else 'posnews'
        raw_text = news[text_column].tolist()

    if training_args.extra_info:
        retrieved_file = training_args.text_file[:-4]+"_retrieved.csv"
        bodies = pd.read_csv(retrieved_file)["retrieved"]
        bodies = bodies.fillna("").tolist()
        raw_text = [t+" "+b for t, b in zip(raw_text, bodies)]

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length-1
    max_source_length = data_args.max_source_length
    padding = "max_length" if data_args.pad_to_max_length else False
    text_inputs = tokenizer(raw_text, padding=padding, max_length=max_source_length,
        truncation=True, return_tensors="pt", return_special_tokens_mask=True)
    text_inputs = add_padding(text_inputs, return_array=True)

    # Prepare user data
    user_dir = training_args.userize_ufeat_path if training_args.userize_ufeat_path else ""
    posnews = pd.read_csv("../results/pens_ghg_own/bart/checkpoint-98000/phg/generated_predictions.txt",
                              sep="\n", header=None)[0].tolist()
    user_df = pd.read_pickle("../datasets/specialize_own/test.pkl")
    user_df["posnews"] = posnews

    ufeat_file = os.path.join(user_dir, "user_features_test.npz")
    user_dict = load_user_features(ufeat_file) if training_args.do_train else None
    editor_news_feat = np.load(os.path.join(user_dir, "editor_headline/news_features.npz"))["news_features"]
    first_news_feat = np.load(os.path.join(user_dir, "first_stage/news_features.npz"))["news_features"]



    for ui in tqdm(range(training_args.userwise_index_sta, training_args.userwise_index_end, 1)):
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomSeq2SeqTrainingArguments))
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

        training_args.output_dir = training_args.output_dir + f"/index{ui}"
        training_args.logging_dir = training_args.output_dir + f"/index{ui}/log"
        os.makedirs(training_args.output_dir, exist_ok=True)

        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            #bart_local_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        # Adjust model_input_names to fit the model
        config.userize = training_args.userize
        config.userize_loss = training_args.userize_loss
        config.userize_mum = training_args.userize_mum
        config.userize_dot = training_args.userize_dot
        config.userize_complex_proj = training_args.userize_complex_proj
        config.userize_type_embedding = training_args.userize_type_embedding
        config.userize_ufeat_type = training_args.userize_ufeat_type

        if training_args.userize:
            config.user_token_length = training_args.userize_user_token_length
            config.decoder_user_token_id = tokenizer.user_token_id

        # Build model
        model = UTCBart.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model.resize_token_embeddings(len(tokenizer))

        if model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        if (
            hasattr(model.config, "max_position_embeddings")
            and model.config.max_position_embeddings < data_args.max_source_length
        ):
            if model_args.resize_position_embeddings is None:
                logger.warning(
                    f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                    f"to {data_args.max_source_length}."
                )
                model.resize_position_embeddings(data_args.max_source_length)
            elif model_args.resize_position_embeddings:
                model.resize_position_embeddings(data_args.max_source_length)
            else:
                raise ValueError(
                    f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                    f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                    "resize the model's position encodings by passing `--resize_position_embeddings`."
                )

        do_predict = training_args.do_predict

        if training_args.do_train or training_args.do_eval:
            # Preprocessing the datasets
            train_dataset, eval_dataset, test_dataset = build_datasets(data_args, training_args, model_args, tokenizer, news, text_inputs, user_df, user_dict, editor_news_feat, first_news_feat, ui)

            # Build trainer
            trainer = build_trainer(model_args, data_args, training_args, model, tokenizer, train_dataset, eval_dataset)

        # Training
        if training_args.do_train:
            logger.info("*** Train ***")
            train_process(training_args, data_args, trainer, train_dataset)
        
        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            eval_process(training_args, data_args, trainer, eval_dataset)

        training_args.do_predict = do_predict
        if training_args.do_predict:
            logger.info("*** Predict ***")
            if training_args.retrieve_online is not True:
                training_args.do_train=False
                training_args.do_eval=False
                training_args.retrieve_online=True
                training_args.predict_with_generate=True
            training_args.eval_with_test_data=True
            training_args.text_file="../results/pens_ghg_own/bart/checkpoint-98000/all_news/generated_predictions.txt"

            _, _, test_dataset = build_datasets(data_args, training_args, model_args, tokenizer, news, text_inputs, user_df, user_dict, editor_news_feat, first_news_feat, ui)

            output_dir = training_args.output_dir
            if training_args.remove_after_predict:
                remove_files(output_dir)

            if training_args.find_best_checkpoint:
                for filename in os.listdir(output_dir):
                    if "checkpoint" in filename:
                        model_args.model_name_or_path = os.path.join(output_dir, filename)
                        training_args.output_dir = model_args.model_name_or_path
                        model.load_state_dict(torch.load(model_args.model_name_or_path+"/pytorch_model.bin"), strict=False)
                        trainer = build_trainer(model_args, data_args, training_args, model, tokenizer, None, None)
                        logger.warning(f"Load model from {model_args.model_name_or_path}")
                        print("Load ", model_args.model_name_or_path)
                        predict_process(training_args, data_args, trainer, test_dataset, tokenizer)
            else:
                training_args.output_dir = model_args.model_name_or_path
                model.load_state_dict(torch.load(model_args.model_name_or_path+"/pytorch_model.bin"), strict=False)
                trainer = build_trainer(model_args, data_args, training_args, model, tokenizer, None, None)
                logger.warning(f"Load model from {model_args.model_name_or_path}")
                predict_process(training_args, data_args, trainer, test_dataset, tokenizer)


if __name__=="__main__":
    main()

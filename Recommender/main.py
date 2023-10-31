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
import torch

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric

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
from config.arguments import ModelArguments, DataTrainingArguments, RecSeq2SeqTrainingArguments
# NEW:
from data.build_datasets import build_datasets
from pipelines.build_trainer import build_trainer

from process import train_process, eval_process, predict_process
from process import get_news_features, get_recommended_score

from model.recommender import BAREC

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


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomSeq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        os.makedirs(training_args.output_dir, exist_ok=True)
        json_file_name = sys.argv[1].split('/')[-1]
        shutil.copyfile(sys.argv[1], "{}/{}".format(training_args.output_dir, json_file_name))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    #log_level = training_args.get_process_log_level()
    log_level = logging.WARNING
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    )

    # Build model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model.resize_token_embeddings(len(tokenizer))
    model = BAREC(model, training_args)
    #ckpt_path = os.path.join(training_args.output_dir, "pytorch_model.bin")
    ckpt_path = os.path.join(model_args.model_name_or_path, "pytorch_model.bin")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        model_info = model.load_state_dict(ckpt, strict=False)
        print("Load the pretrained model from {model_args.model_name_or_path}")
        print(model_info)


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
    # Preprocessing the datasets
    train_dataset, eval_dataset, test_dataset = build_datasets(data_args, training_args, model_args, tokenizer)

    # Build trainer
    trainer = build_trainer(model_args, data_args, training_args, model, tokenizer, train_dataset, eval_dataset)

    # Training
    if training_args.do_train:
        logger.info("*** Train ***")
        train_process(training_args, data_args, trainer, train_dataset)
    
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        eval_process(training_args, data_args, trainer, eval_dataset)

    if training_args.do_predict:
        if training_args.get_news_features:
            get_news_features(training_args, trainer, test_dataset)
            logger.info("*** Predict to get news features ***")

        elif training_args.get_recommended_score:
            logger.info("*** Predict to get recommended score ***")
            get_recommended_score(training_args, trainer, test_dataset)

        #elif training_args.get_user_features:
        #    # TODO: to implement
        #    logger.info("*** Predict to get user features ***")
        #    get_user_features(training_args, trainer, test_dataset)
            
        else:
            predict_process(training_args, data_args, trainer, test_dataset, tokenizer)
            logger.info("*** Predict ***")

    # Save model card 
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)

if __name__=="__main__":
    main()

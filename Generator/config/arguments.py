from dataclasses import dataclass, field
from typing import Optional, List
from transformers import Seq2SeqTrainingArguments


@dataclass
class CustomSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    developing: bool = field(
        default=False,
        metadata={
            "help": "Developing"
        },
    )
    train_only: str = field(
        default=None,
        metadata={
            "help": "Only Train layers containing specify names"
        },
    )
    text_file: str = field(
        default=None,
        metadata={
            "help": "Text input, a csv file should be the same size as news.pkl"
        },
    )
    do_finetune: bool = field(
        default=False,
        metadata={
            "help": "Whether to finetune on test data"
        },
    )
    find_best_checkpoint: bool = field(
        default=False,
        metadata={
            "help": "Find the best checkpoint and reassign the model_name_or_path and output_dir"
        },
    )
    remove_after_predict: bool = field(
        default=False,
        metadata={
            "help": "Remove the model files after prediction"
        },
    )
    save_model_accord_to_rouge: bool = field(
        default=False,
        metadata={
            "help": "Whether to save model according to ROUGE-1 score instead of loss."
        },
    )
    shuffle_before_select: bool = field(
        default=True,
        metadata={
            "help": "Whether to shuffle the datasets"
        },
    )
    eval_with_test_data: bool = field(
        default=False,
        metadata={
            "help": "Evaluate by the personal headlines, personal headlines are only included in test.csv"
        },
    )
    predict_txt_file: str = field(
        default=None,
        metadata={
            "help": "File path of the prediction results. Used in build_datasets.py to build the input. If not given, take the GT headline as input."
        },
    )

    mlm: bool = field(
        default=False,
        metadata={
            "help": "Whether to do MLM."
        },
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={
            "help": "Probability of MLM, dedault=0.15"
        },
    )
    mlm_update_all: bool = field(
        default=False,
        metadata={
            "help": "Whether to update the full sequence or just the perturbation"
        },
    )

    userize: bool = field(
        default=False,
        metadata={
            "help": "Used in preprocess_function() and model initialization, user_features will be built if userize is True. Besides, the attention_mask will be prolonged if user_features is built, which is controlled in collator function."
        },
    )
    userize_ufeat_path: str = field(
        default=None,
        metadata={
            "help": "The directory to load the preprocessed user features"
        }
    )
    userize_user_token_length: int = field(
        default=5,
        metadata={
            "help": "The projection size of user features"
        }
    )
    userize_loss: bool = field(
        default=False,
        metadata={
            "help": "Whether to train with Mask User Modeling"
        }
    )
    userize_mum: bool = field(
        default=False,
        metadata={
            "help": "Whether mask the user embeddings, if mask user embedding, we would only update the masked part"
        }
    )
    userize_ufeat_type: str = field(
        default="closest_news",
        metadata={
            "help": "The ways to construct user feature. Supported type: closest, average, tail_feat, user_feat, text_closest. Related arguments: (tail_feat, userize_ctr_threshold), (text_closest, turn_off userize_mum and userize_pum), userize_user_token_length"
        }
    )
    userize_complex_proj: bool = field(
        default=False,
        metadata={
            "help": "The projection layer for user features"
        }
    )
    userize_type_embedding: bool = field(
        default=False,
        metadata={
            "help": "Whether to add the type embedding to user tokens"
        }
    )
    userize_ctr_threshold: int = field(
        default=-1,
        metadata={
            "help": "The threshold between head news and tail news for CTR distribution"
        }
    )
    userize_ctr_quant: float = field(
        default=0.5,
        metadata={
            "help": "The threshold quantile btween head news and tail news for CTR distribution for each user"
        }
    )
    userize_dot: bool = field(
        default=False,
        metadata={
            "help": "Whether to optimize personalization by dot product between user feature and generated headline"
        }
    )

    extra_info: bool = field(
        default=False,
        metadata={
            "help": "Whether add extra information to input, "
        }
    )
    retrieve_online: bool = field(
        default=False,
        metadata={
            "help": "retrieve sentences from body onlinely. For finetuning and testing, this should be set to True"
        },
    )

    userwise: bool = field(
        default=False,
        metadata={
            "help": "whether to train the model user-wisely",
        },
    )
    userwise_index: int = field(
        default=None,
        metadata={
            "help": "User index. between 0 to 102",
        },
    )
    userwise_split: str = field(
        default=None,
        metadata={
            "help": "format: `num_train num_eval num_test` where they should add up to 200",
        },
    m
    userwise_seed: int = field(
        default=None,
        metadata={
            "help": "random seed for shuffling in build_dataset",
        },
    )
    userwise_sample_type: str = field(
        default='random',
        metadata={
            "help": "how to get the userwise training data, default='random'",
        },
    )
    # Not Use
    corrupt: bool = field(
        default=False,
        metadata={
            "help": "Used in preprocess_function()"
        },
    )
    corrupt_token_infilling: bool = field(
        default=False,
        metadata={
            "help": "Used in preprocess_function()"
        },
    )
    corrupt_token_shuffle: bool = field(
        default=False,
        metadata={
            "help": "Used in preprocess_function()"
        },
    )
    corrupt_max_mask_num: int = field(
        default=1,
        metadata={
            "help": "Used in preprocess_function(), max masking number"
        },
    )
    corrupt_max_mask_length_lambda: int = field(
        default=3,
        metadata={
            "help": "Used in preprocess_function(), lambda in poisson distribution to control the max masking length"
        },
    )
    userize_pum: bool = field(
        default=False,
        metadata={
            "help": "Prompt user modeling, mse loss for the <user> token"
        }
    )
    userwise_index_sta: int = field(
        default=None,
        metadata={
            "help": "starting user index. Between 0 to 102",
        },
    )
    userwise_index_end: int = field(
        default=None,
        metadata={
            "help": "Ending user index. Between 0 to 102",
        },
    )

    

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
            "the model's position embeddings."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                #assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                #assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length

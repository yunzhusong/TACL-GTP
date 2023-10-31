import pdb
import nltk
import numpy as np
from datasets import load_metric
from transformers import DataCollatorForSeq2Seq
from pipelines.trainer import Trainer
from pipelines.utils import metric_recommend

def build_trainer(model_args, data_args, training_args, model, tokenizer, train_dataset, eval_dataset):
    model.args = training_args

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    def compute_metrics(eval_preds, eval_gen=False):
        # For recommendation system
        preds = eval_preds.predictions[0] if isinstance(eval_preds.predictions, tuple) else eval_preds.predictions
        labels = eval_preds.label_ids
        result = metric_recommend(preds, labels)
        print("Pred:  ", preds[0][:10])
        print("Label: ", labels[0][:10])

        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()

        result = {k: round(v, 4) for k, v in result.items()}
        return result


    # Initialize our Trainer
    trainer = Trainer(
        model=model,
	args=training_args,
	model_args=model_args,
	data_args=data_args,
	train_dataset=train_dataset if training_args.do_train else None,
	eval_dataset=eval_dataset if training_args.do_eval else None,
	tokenizer=tokenizer,
	data_collator=data_collator,
	compute_metrics=compute_metrics,
    )

    return trainer

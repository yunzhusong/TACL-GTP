import pdb
from tqdm import tqdm
import numpy as np
from pipelines.trainer_base import BaseTrainer

#// For predction_step()
import torch
from torch import nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.trainer_pt_utils import (
    nested_detach
)

class Trainer(BaseTrainer):
    def __init__(
        self,
        model=None,
        args=None,
        model_args=None,
        data_args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
    ):
        BaseTrainer.__init__(
            self, model, args, model_args, data_args, data_collator, train_dataset,
            eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        self.label_names = ["labels_rec"]

        has_labels = all(inputs.get(k) is not None for k in self.label_names)

        # NEW: when sample size is not two, we don't calculating loss
        is_two_samples = False
        if has_labels:
            is_two_samples = inputs.get(self.label_names[0]).shape[1] == 2

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels and is_two_samples:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels) 

    def news_features(self, dataset):

        self._memory_tracker.start()
        dataloader = self.get_eval_dataloader(dataset)
        self.model.eval()

        news_features = []
        for step, inputs in tqdm(enumerate(dataloader)):
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            bs = inputs["input_ids"].shape[0]
            seq_len = input_ids.shape[-1]
            input_ids = input_ids.reshape(bs, -1, seq_len).to(self.model.device)
            attention_mask = attention_mask.reshape(bs, -1, seq_len).to(self.model.device)
            news_features.append(
                self.model.news_encoder(
                    input_ids,
                    attention_mask
                ).detach().cpu()
            )
        all_news_features = np.concatenate(news_features).squeeze()
        return all_news_features

    def predict_recommended_score(self, dataset):

        self._memory_tracker.start()
        dataloader = self.get_eval_dataloader(dataset)
        self.model.eval()

        scores = []
        for step, inputs in tqdm(enumerate(dataloader)):
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            bs = inputs["input_ids"].shape[0]
            seq_len = input_ids.shape[-1]
            input_ids = input_ids.reshape(bs, -1, seq_len).to(self.model.device)
            attention_mask = attention_mask.reshape(bs, -1, seq_len).to(self.model.device)
            user_input_ids = inputs["user_input_ids"].to(self.model.device)
            user_attention_mask = inputs["user_attention_mask"].to(self.model.device)
            scores.append(
                self.model(
                    input_ids,
                    attention_mask,
                    user_input_ids=user_input_ids,
                    user_attention_mask=user_attention_mask,
                )["scores"].detach().cpu()
            )
        all_scores = np.concatenate(scores).squeeze()
        return all_scores

    def compute_loss(self, model, inputs, return_outputs=False):
        
        if self.label_smoother is not None and "labels_rec" in inputs:
            labels_rec = inputs.pop("labels_rec")
        else:
            labels_rec = None

        outputs = model(**inputs)

        loss = outputs["loss"]

        if model.training:
            if outputs["rec_loss"] is not None:
                self.our_logs["rec_loss"].append(outputs["rec_loss"].item())

            if outputs["ctr_loss"] is not None:
                self.our_logs["ctr_loss"].append(outputs["ctr_loss"].item())

        return (loss, outputs) if return_outputs else loss 



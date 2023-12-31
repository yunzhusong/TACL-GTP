import os
import pdb
# For __init__
import torch
from torch import nn as nn
from transformers import Seq2SeqTrainer
from transformers.trainer_callback import TrainerCallback
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from typing import Optional, Union, Dict, Any, Callable, Tuple, List
import shutil
from transformers.utils import logging
logger = logging.get_logger(__name__)

class BaseSeq2SeqTrainer(Seq2SeqTrainer):
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
        Seq2SeqTrainer.__init__(self, model, args, data_collator, train_dataset,
                                eval_dataset, tokenizer, model_init,
                                compute_metrics, callbacks, optimizers)
        self.model_args = model_args
        self.data_args = data_args

        # Make specified parameters trainable
        self._freeze_all_params(self.model)
        self._unfreeze_specified_params(self.model, train_only=args.train_only)

        if args.userize_dot:
            print("Fix recommender")
            for param in model.recommender.parameters():
                param.requires_grad = False

        # Show number of parameters
        all_param_num = sum([p.nelement() for p in self.model.parameters()])
        trainable_param_num = sum([
            p.nelement()
            for p in self.model.parameters()
            if p.requires_grad == True
        ])
        print(f"All parameters : {all_param_num}")
        print(f"Trainable parameters : {trainable_param_num}")

        # For best model saving
        self._ckpt_eval_loss = {}
        if self.args.save_model_accord_to_rouge:
            self._ckpt_eval_rouge = {}

    def _freeze_all_params(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def _unfreeze_specified_params(self, model, train_only=None):
        if train_only is not None:
            logger.warning("Train partial network")
            names = train_only.split()
            for train_name in names:
                for name, sub_module in self.model.named_modules():
                    if train_name in name:
                        for param in sub_module.parameters():
                            param.requires_grad = True
        else:
            for param in model.parameters():
                param.requires_grad = True

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, metrics)

            # NEW: record metric
            if self.args.save_model_accord_to_rouge:
                #self._cur_eval_rouge = metrics['eval_rouge1']
                self._cur_eval_rouge = metrics['eval_combined_score']
            self._cur_eval_loss = metrics['eval_loss'] if 'eval_loss' in metrics else 0

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
 
    def _rotate_checkpoints(self, use_mtime=False, output_dir=None) -> None:
        """
        Modification:
            - record eval loss/rouge and maintain best model

        NOTE: to make this function works properly,
              the save_steps should be multiples of evaluation_steps
        """
        # NEW
        if self.args.eval_steps != self.args.save_steps:
            raise Exception(
                "To properly store best models, please make sure eval_steps equals to save_steps."
            )

        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Sort according to evaluation steps
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime,
                                                      output_dir=output_dir)

        # NEW: record the eval metric for the last checkpoint
        self._ckpt_eval_loss[checkpoints_sorted[-1]] = self._cur_eval_loss
        if self.args.save_model_accord_to_rouge:
            self._ckpt_eval_rouge[checkpoints_sorted[-1]] = self._cur_eval_rouge

        # Check if we should delete older checkpoint(s)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        # NEW: sort according to metrics (descending for loss)
        checkpoints_sorted = [
            k for k, v in sorted(
                self._ckpt_eval_loss.items(), key=lambda x: x[1], reverse=True)
        ]

        number_of_checkpoints_to_delete = max(
            0,
            len(checkpoints_sorted) - self.args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:
                                                       number_of_checkpoints_to_delete]

        # NEW: sort according to metrics (ascending for rouge)
        if self.args.save_model_accord_to_rouge:
            checkpoints_sorted_rouge = [
                k for k, v in sorted(self._ckpt_eval_rouge.items(),
                                     key=lambda x: x[1],
                                     reverse=False)
            ]
            checkpoints_to_be_deleted_rouge = checkpoints_sorted_rouge[:
                                                                       number_of_checkpoints_to_delete]
            # Only delete the intersect checkpoints
            checkpoints_to_be_deleted = list(
                set(checkpoints_to_be_deleted).intersection(
                    set(checkpoints_to_be_deleted_rouge)))

        for checkpoint in checkpoints_to_be_deleted:
            logger.info(
                "Deleting older checkpoint [{}] due to args.save_total_limit".
                format(checkpoint))
            del self._ckpt_eval_loss[checkpoint]  # NEW: remove the delted ckpt
            if self.args.save_model_accord_to_rouge:
                del self._ckpt_eval_rouge[checkpoint]  # NEW: remove the delted ckpt
            try:
                shutil.rmtree(checkpoint)
            except:
                print("Checkpoint is already delted.")

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        Modification:
            - Also record model_args and data_args
        """
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            if isinstance(unwrap_model(self.model), PreTrainedModel):
                if state_dict is None:
                    state_dict = self.model.state_dict()
                unwrap_model(self.model).save_pretrained(output_dir,
                                                         state_dict=state_dict)
            else:
                logger.info(
                    "Trainer.model is not a `PreTrainedModel`, only saving its state dict."
                )
                if state_dict is None:
                    state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        torch.save(self.model_args, os.path.join(output_dir,
                                                 "model_args.bin"))  # NEW
        torch.save(self.data_args, os.path.join(output_dir,
                                                "data_args.bin"))  # NEW

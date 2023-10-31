import pdb
from pipelines.trainer_seq2seq_base import BaseSeq2SeqTrainer


class Trainer(BaseSeq2SeqTrainer):
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
        BaseSeq2SeqTrainer.__init__(
            self, model, args, model_args, data_args, data_collator, train_dataset,
            eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers)

    def _prepare_inputs(self, inputs):
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        #if self.args.userize and self.args.userize_loss:
        if self.args.userize:

            # NOTE: dot loss will use the user_features
            #user_features = inputs.pop("user_features")
            user_features = inputs["user_features"]

            inputs["user_embeds"] = self.model.model.encoder.get_user_embeds(user_features)
 

        return inputs

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs, dot_loss=self.args.userize_dot)

        mum_loss = None
        if self.args.userize and self.args.userize_loss:
            inputs.pop("labels")

            #if self.args.userize_dot:
            #    dot_loss = model(**inputs, dot_loss=True)["loss"]

            if self.args.userize_mum:
                # if MUM, we would mask the user embeds and only update the masked embedding
                inputs["user_labels"] = inputs["user_embeds"]
                inputs["user_embeds"], inputs["user_mask"] = model.mask_user_modeling(inputs["user_embeds"])
                mum_loss = model(**inputs, mum_loss=True)["loss"]

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if mum_loss is not None:
            loss += mum_loss

        return (loss, outputs) if return_outputs else loss

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / 
                (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            # NEW: record additional loss
            if self.args.userize and self.args.userize_loss:

                if self.args.userize_mum:
                    tr_loss_mum_scalar = self.model.tr_loss_mum.item()
                    # reset tr_loss_reg to zero
                    self.model.tr_loss_mum -= self.model.tr_loss_mum
    
                    logs["loss_mum"] = round(
                        tr_loss_mum_scalar /
                        (self.state.global_step - self._globalstep_last_logged), 4)

                if self.args.userize_dot:
                    tr_loss_mum_scalar = self.model.tr_loss_dot.item()
                    # reset tr_loss_reg to zero
                    self.model.tr_loss_dot -= self.model.tr_loss_dot
    
                    logs["loss_dot"] = round(
                        tr_loss_mum_scalar /
                        (self.state.global_step - self._globalstep_last_logged), 4)
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
 

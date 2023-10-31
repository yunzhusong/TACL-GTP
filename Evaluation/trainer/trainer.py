""" Overide the Seq2SeqTrainer to support the user_feature inputs """
from transformers import Seq2SeqTrainer
from transformers.utils import logging, is_torch_tpu_available
from transformers import GenerationConfig

logger = logging.get_logger(__name__)

log_level = getattr(logging, "WARNING")
logger.setLevel(log_level)

class CustomizedSeq2SeqTrainer(Seq2SeqTrainer):

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", **gen_kwargs):
        """
        Add generation config before generation function to avoid repeatly logging information
        """
        if gen_kwargs.get("generation_config") is None:
            new_gen_kwargs = {
                "generation_config": GenerationConfig.from_model_config(self.model.config)
            }
            gen_kwargs.update(new_gen_kwargs)

        return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix, **gen_kwargs)


    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix="test", **gen_kwargs):
        """
        Add generation config before generation function to avoid repeatly logging information
        """
        if gen_kwargs.get("generation_config") is None:
            new_gen_kwargs = {
                "generation_config": GenerationConfig.from_model_config(self.model.config)
            }
            gen_kwargs.update(new_gen_kwargs)

        return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix, **gen_kwargs)


    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """ This function pre-process the inputs before prediction.
        Get the inputs_embeds from input_ids and user_features.
        """
        if model.config.userize:
            inputs = super()._prepare_inputs(inputs)

            user_embeds = model.user_embedding(inputs.pop("user_features"))
            inputs["inputs_embeds"] = model._insert_user_features(inputs.pop("input_ids"), user_embeds)
            inputs["attention_mask"] = model._prolong_mask(inputs.pop("attention_mask"), model.config.num_user_tokens)

        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / 
                (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            if self.args.userize and self.args.rec_loss:
                tr_loss_rec_scalar = self.model.tr_loss_rec.item()
                self.model.tr_loss_rec -= self.model.tr_loss_rec
                logs["loss_rec"] = round(tr_loss_rec_scalar /
                    (self.state.global_step - self._globalstep_last_logged), 4)

            if self.args.userize and self.args.mum:
                tr_loss_mum_scalar = self.model.tr_loss_mum.item()
                self.model.tr_loss_mum -= self.model.tr_loss_mum
                logs["loss_mum"] = round(tr_loss_mum_scalar /
                    (self.state.global_step - self._globalstep_last_logged), 4)

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

""" PyTorch BART model."""
import pdb
import random
import numpy as np
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.utils import logging
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import (
    BartEncoder, BartDecoder, BartModel, BartForConditionalGeneration
)
from transformers.models.bart.modeling_bart import _expand_mask, shift_tokens_right

from model.modules import SelfAttention

logger = logging.get_logger(__name__)


class UTCBart(BartForConditionalGeneration):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        if config.userize:
            self.model = UTCBartModel(config)
        else:
            self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))

        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # NEW: for Masked User Modaling
        if config.userize and config.userize_loss:
            if config.userize_mum:
                self.user_head = nn.Linear(config.d_model, config.d_model)
                self.tr_loss_mum = torch.tensor(0.0).to(self.device)
            if config.userize_dot:
                self.tr_loss_dot = torch.tensor(0.0).to(self.device)

        # Initialize weights and apply final processing
        self.init_weights()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        user_features: Optional[torch.FloatTensor] = None, # NEW
        user_embeds: Optional[torch.FloatTensor] = None, # NEW
        mum_loss: Optional[bool] = None,
        dot_loss: Optional[bool] = None,
        unmask_user_embeds: Optional[torch.FloatTensor] = None, # NEW
        user_labels: Optional[torch.FloatTensor] = None, # NEW
        user_mask: Optional[torch.Tensor] = None, # NEW
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        user_labels (torch.FloatTensor)
            for MUM loss: it is masked user token embedding
            for PUM loss: it is the projected user feature on word space
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        model_inputs = {
            "input_ids":input_ids,
            "attention_mask":attention_mask,
            "decoder_input_ids":decoder_input_ids,
            "encoder_outputs": encoder_outputs,
            "decoder_attention_mask":decoder_attention_mask,
            "head_mask":head_mask,
            "decoder_head_mask":decoder_head_mask,
            "cross_attn_head_mask":cross_attn_head_mask,
            "past_key_values":past_key_values,
            "inputs_embeds":inputs_embeds,
            "decoder_inputs_embeds":decoder_inputs_embeds,
            "use_cache":use_cache,
            "output_attentions":output_attentions,
            "output_hidden_states":output_hidden_states,
            "return_dict":return_dict,
        }
        if self.config.userize:
            model_inputs["user_features"] = user_features
            model_inputs["user_embeds"] = user_embeds

        outputs = self.model(**model_inputs)

        lm_logits_base = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            #// Generation Loss
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits_base.reshape(-1, self.config.vocab_size), labels.reshape(-1))

            if dot_loss:
                #// DOT Loss
                labels_mask = (labels!=-100).to(torch.int)
                news_feat = self.recommender.news_encoder(inputs_embeds=outputs[0], attention_mask=labels_mask)
                cos_fct = nn.CosineEmbeddingLoss()
                target = torch.ones(news_feat.shape[0]).to(self.device)
                loss_dot = cos_fct(news_feat.squeeze(1), user_features, target)
                #loss_dot = -torch.mean(torch.bmm(news_feat, user_features.unsqueeze(-1)))
                self.tr_loss_dot += loss_dot.item()
                masked_lm_loss += loss_dot

        if mum_loss:
            #// MUM loss
            pred_user_embeds = self.user_head(outputs["encoder_last_hidden_state"][:,:self.config.user_token_length]) #NOTE:v3
            loss_mse = MSELoss(reduce=user_mask is None)
            masked_lm_loss = loss_mse(pred_user_embeds, user_labels)
            if user_mask is not None:
                # ensure the loss value is large enough
                masked_lm_loss = torch.mean(torch.sum(masked_lm_loss[torch.arange(len(user_mask)), user_mask], dim=1))
            self.tr_loss_mum += masked_lm_loss.item()

        if not return_dict:
            output = (lm_logits_base,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        output = Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits_base,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

        return output

    def _prolong_mask(self, attention_mask, length):
        """
        Prolong the `attention mask` with size `length`
        """
        extended_mask = torch.ones((attention_mask.size(0), length)).to(attention_mask)
        attention_mask = torch.cat((extended_mask, attention_mask), dim=1)
        return attention_mask

    def mask_user_modeling(
        self,
        user_embeds,
    ):
        mask_embed = self.model.shared.weight[-1]
        mask_index = torch.randint(0, user_embeds.size(1), (user_embeds.size(0),))
        user_embeds[torch.arange(len(mask_index)), mask_index] = mask_embed
        #mask = torch.zeros(user_embeds.shape[:2], dtype=int)
        #mask.scatter_(1, mask_index.reshape(-1,1), 1)

        return user_embeds, mask_index

    def _prepare_decoder_input_ids_for_generation(
        self, input_ids: torch.LongTensor, decoder_start_token_id: int = None, bos_token_id: int = None
    ) -> torch.LongTensor:

        decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        decoder_input_ids = (
            torch.ones((input_ids.shape[0], 1), dtype=torch.long, device=input_ids.device) * decoder_start_token_id
        )
        return decoder_input_ids

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        decoder_input_ids = labels

        decoder_input_ids = shift_tokens_right(decoder_input_ids, self.config.pad_token_id, self.config.decoder_start_token_id)
        return decoder_input_ids

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, input_ids: torch.LongTensor, model_kwargs
    ):
        if "encoder_outputs" not in model_kwargs:
            # retrieve encoder hidden states
            encoder = self.get_encoder()
            encoder_kwargs = {
                argument: value
                for argument, value in model_kwargs.items()
                if not (argument.startswith("decoder_") or argument.startswith("cross_attn"))
            }
            model_kwargs["encoder_outputs"]: ModelOutput = encoder(input_ids, return_dict=True, **encoder_kwargs)
        return model_kwargs


class UTCBartModel(BartModel):

    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = UTCBartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        # Initialize weights and apply final processing
        self.init_weights()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        user_features: Optional[torch.FloatTensor] = None,
        user_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                user_features=user_features,
                user_embeds=user_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class UTCBartEncoder(BartEncoder):

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding]=None):
        super().__init__(config, embed_tokens)

        if config.userize_complex_proj:
            self.user_proj = SelfAttention(config.d_model, 6, dropout=0.2, user_token_len=config.user_token_length)

        else:
            n_centroid = 1
            if "centroid" in config.userize_ufeat_type:
                n_centroid = config.user_token_length
            
            self.user_proj = nn.Linear(config.d_model*n_centroid, config.d_model*config.user_token_length)

        if config.userize_type_embedding:
            self.token_type_embeddings = nn.Embedding(2, config.d_model)
        self.add_type_embedding = config.userize and config.userize_type_embedding


    def get_user_embeds(
        self,
        user_features,
    ):
        
        if sum(user_features[:,0]==0)==len(user_features):
            #// If we use the raw text as user_features
            user_embeds = self.embed_tokens(user_features) * self.embed_scale
            return user_embeds

        #// If we use a preprocess vectors as user_features
        if len(user_features.shape)==3:
            user_features = user_features.reshape(user_features.shape[0], -1)
        with torch.no_grad():
            user_embeds = self.user_proj(user_features)
        user_embeds = torch.stack(user_embeds.chunk(self.config.user_token_length, -1), 1)
        return user_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        user_features: Optional[torch.FloatTensor] = None,
        user_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        if user_embeds is None and user_features is None:
            raise ValueError("Please specify user_embeds or user_features")
        elif user_embeds is None:
            user_embeds = self.get_user_embeds(user_features)

        if self.add_type_embedding:
            inputs_embeds = torch.cat((
                user_embeds + self.token_type_embeddings(torch.zeros(user_embeds.shape[:-1]).to(attention_mask)),
                inputs_embeds + self.token_type_embeddings(torch.ones(inputs_embeds.shape[:-1]).to(attention_mask))
            ), dim=1)

        else:
            inputs_embeds = torch.cat((user_embeds, inputs_embeds), dim=1)
        input_shape = inputs_embeds.size()[:-1]

        #assert input_shape==attention_mask.shape

        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


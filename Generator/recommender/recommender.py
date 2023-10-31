import pdb
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F

from transformers import BartForConditionalGeneration
from recommender.modules import AttentionPooling, CTRRegressionHead, initialize_weight
from transformers.file_utils import ModelOutput

class BAREC(BartForConditionalGeneration):

    def __init__(self, model: BartForConditionalGeneration, args):
        super().__init__(model.config)
        self.args = args

        # Load initial weights
        self.load_state_dict(model.state_dict())

        self.title_attn_pool = AttentionPooling(model.config.d_model, model.config.d_model)
        self.user_attn_pool = AttentionPooling(model.config.d_model, model.config.d_model)
        self.title_attn_pool.apply(initialize_weight)
        self.user_attn_pool.apply(initialize_weight)
        self.rec_weight = args.rec_weight

        self.ctr_regression_head = None
        if args.ctr_loss:
            self.ctr_regression_head = CTRRegressionHead(model.config.d_model, model.config.d_model)
            self.ctr_regression_head.apply(initialize_weight)
            self.ctr_weight = args.ctr_weight


    def news_encoder(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        ):
        """
        input_ids: (batch_size, candidate_size, seq_len)
        """

        if input_ids is not None:
            batch_size, candidate_size, seq_len = input_ids.shape
            input_ids = input_ids.reshape(-1, seq_len)
            attention_mask = attention_mask.reshape(-1, seq_len)
        elif inputs_embeds is not None:
            candidate_size = 1
            batch_size, seq_len, _ = inputs_embeds.shape

        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        news_vec = self.title_attn_pool(encoder_outputs[0], attention_mask.reshape(-1, seq_len))
        news_vec = news_vec.reshape(batch_size, candidate_size, -1)
        return news_vec


    def user_encoder(
        self,
        user_news_features,
        attn_mask=None,
        sample_news_features=None,
    ):
        if sample_news_features is not None:
            doc_product = torch.bmm(user_news_features, sample_news_features.permute(0,2,1))
            weight = doc_product / (torch.norm(user_news_features, p=2, dim=-1).unsqueeze(-1) + 1e-6)
            weight = weight / (torch.norm(sample_news_features, p=2, dim=-1).unsqueeze(1) + 1e-6)

            user_features = []
            num_samples = sample_news_features.shape[1]
            for i in range(num_samples):
                user_features.append(self.user_attn_pool(user_news_features, attn_mask=attn_mask*weight[:,:,i]))
            user_features = torch.stack(user_features, dim=1)
        else:
            user_features = self.user_attn_pool(user_news_features, attn_mask=attn_mask)

        return user_features

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        user_features=None, # NEW
        user_input_ids=None, # NEW
        user_attention_mask=None, # NEW
        labels_rec=None, # NEW
        labels_ctr=None, # NEW
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        is_two_samples = True if labels_rec is not None and labels_rec.shape[1]==2 else False

        #// Reshape for recommendation
        seq_len = input_ids.shape[-1]
        batch_size = user_input_ids.shape[0]
        input_ids = input_ids.reshape(batch_size, -1, seq_len)
        attention_mask = attention_mask.reshape(batch_size, -1, seq_len) 

        #/// Encode news for candidate news
        sample_news_vec = self.news_encoder(input_ids, attention_mask)

        #// Encoder news for user history
        user_news_vec = self.news_encoder(user_input_ids, user_attention_mask)
        pad_user_attn_mask = (user_attention_mask[:,:,0]==1).to(user_attention_mask.dtype)

        #// Sample awared
        #user_vec = self.user_encoder(user_news_vec, pad_user_attn_mask, sample_news_vec)
        #scores = torch.bmm(sample_news_vec, user_vec.permute(0,2,1))
        #scores = torch.stack((scores[:,0,0], scores[:,1,1]), dim=1)
        #// Not sample awared
        user_vec = self.user_encoder(user_news_vec, pad_user_attn_mask)
        user_vec = user_vec.unsqueeze(-1)
        scores = torch.bmm(sample_news_vec, user_vec).squeeze()

        pred_ctr = None
        if self.ctr_regression_head is not None:
            pred_ctr = self.ctr_regression_head(sample_news_vec)

        loss, ctr_loss, rec_loss = None, None, None

        if is_two_samples:
            #// Recommendation Loss
            loss_fct = CrossEntropyLoss()
            _labels = torch.where(labels_rec==1)[1].to(scores.device)
            rec_loss = loss_fct(scores, _labels)

            if pred_ctr is not None and labels_ctr is not None:
            #// CTR Regression Loss
                loss_fct = MSELoss()
                ctr_loss = loss_fct(pred_ctr.view(-1), labels_ctr.view(-1))
                loss = self.rec_weight * rec_loss + self.ctr_weight * ctr_loss
            else:
                loss = self.rec_weight * rec_loss

        if not return_dict:
            output = (score,)
            return ((loss,) + output) if loss is not None else output

        return ModelOutput(
            loss=loss,
            scores=scores,
            pred_ctr=pred_ctr,
            user_vec=user_vec,
            rec_loss=rec_loss,
            ctr_loss=ctr_loss,
        )


import urllib.parse
from argparse import ArgumentParser

import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F

from torchtext.vocab import GloVe, vocab

class TransformerBottleAutoEncoder(LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr

        # Load GloVe
        glove = GloVe(name="6B", dim=100, max_vectors=50000)

        # Embedding
        special_tokens = ["<unk>", "<s>", "</s>", "<pad>"]
        glove_embeds = glove.vectors
        special_token_embeds = torch.empty((len(special_tokens), glove_embeds.shape[1]))
        nn.init.normal_(special_token_embeds)
        self.embedding = nn.Embedding.from_pretrained(
            torch.concat([special_token_embeds, glove_embeds]))

        # Tie embedding and language model head
        self.vocab_size, self.embed_dim = self.embedding.weight.shape
        self.lm_head = nn.Linear(self.embed_dim, self.vocab_size)
        self.lm_head.weight = self.embedding.weight

        self.encoder = nn.TransformerEncoderLayer(d_model=100, nhead=2,
                                                  batch_first=True)
        self.decoder = nn.TransformerEncoderLayer(d_model=100, nhead=2,
                                                  batch_first=True)
        self.bottleneck_down = nn.Linear(100, 25)
        self.bottleneck_up = nn.Linear(25, 100)

    def infer(self, input_ids, attention_mask):
        embeds = self.embedding(input_ids)
        z = self.encoder(embeds)
        z = self.bottleneck_down(z)
        z = self.bottleneck_up(z)
        x_hat = self.decoder(z)
        logits = self.lm_head(x_hat)
        return logits

    def bottleneck(self, input_ids, attention_mask):
        embeds = self.embedding(input_ids)
        z = self.encoder(embeds)
        z = self.bottleneck_down(z)
        z = self.bottleneck_up(z)
        return z

    def step(self, batch, batch_idx):
        input_ids, attention_mask = batch
        label_ids = input_ids*attention_mask + (attention_mask-1)*100

        logits = self.infer(input_ids, attention_mask)

        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            label_ids.view(-1),
            ignore_index=-100,
        )
        return loss, {"loss": loss}

    def score(self, batch, batch_idx):
        input_ids, attention_mask = batch
        label_ids = input_ids*attention_mask + (attention_mask-1)*100

        logits = self.infer(input_ids, attention_mask)

        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            label_ids.view(-1),
            ignore_index=-100,
            reduction="none",
        )
        score = loss.view(-1, logits.shape[1]).mean(-1)
        return score

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def predict_step(self, batch, batch_idx):
        loss = self.score(batch, batch_idx)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class TransformerAutoEncoder(LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr

        # Load GloVe
        glove = GloVe(name="6B", dim=100, max_vectors=50000)

        # Embedding
        special_tokens = ["<unk>", "<s>", "</s>", "<pad>"]
        glove_embeds = glove.vectors
        special_token_embeds = torch.empty((len(special_tokens), glove_embeds.shape[1]))
        nn.init.normal_(special_token_embeds)
        self.embedding = nn.Embedding.from_pretrained(
            torch.concat([special_token_embeds, glove_embeds]))

        # Tie embedding and language model head
        self.vocab_size, self.embed_dim = self.embedding.weight.shape
        self.lm_head = nn.Linear(self.embed_dim, self.vocab_size)
        self.lm_head.weight = self.embedding.weight

        self.encoder = nn.TransformerEncoderLayer(d_model=100, nhead=2,
                                                  batch_first=True)
        self.decoder = nn.TransformerEncoderLayer(d_model=100, nhead=2,
                                                  batch_first=True)

    def infer(self, input_ids, attention_mask):
        embeds = self.embedding(input_ids)
        z = self.encoder(embeds)
        x_hat = self.decoder(z)
        logits = self.lm_head(x_hat)
        return logits

    def step(self, batch, batch_idx):
        input_ids, attention_mask = batch
        label_ids = input_ids*attention_mask + (attention_mask-1)*100

        logits = self.infer(input_ids, attention_mask)

        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            label_ids.view(-1),
            ignore_index=-100,
        )
        return loss, {"loss": loss}

    def score(self, batch, batch_idx):
        input_ids, attention_mask = batch
        label_ids = input_ids*attention_mask + (attention_mask-1)*100

        logits = self.infer(input_ids, attention_mask)

        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            label_ids.view(-1),
            ignore_index=-100,
            reduction="none",
        )
        score = loss.view(-1, logits.shape[1]).mean(-1)
        return score

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def predict_step(self, batch, batch_idx):
        loss = self.score(batch, batch_idx)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class LSTMBottleAutoEncoder(LightningModule):
    def __init__(
        self,
        lr: float = 1e-2,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr

        # Load GloVe
        glove = GloVe(name="6B", dim=100, max_vectors=50000)

        # Embedding
        special_tokens = ["<unk>", "<s>", "</s>", "<pad>"]
        glove_embeds = glove.vectors
        special_token_embeds = torch.empty((len(special_tokens), glove_embeds.shape[1]))
        nn.init.normal_(special_token_embeds)
        self.embedding = nn.Embedding.from_pretrained(
            torch.concat([special_token_embeds, glove_embeds]))

        # Tie embedding and language model head
        self.vocab_size, self.embed_dim = self.embedding.weight.shape
        self.lm_head = nn.Linear(self.embed_dim, self.vocab_size)
        self.lm_head.weight = self.embedding.weight

        self.encoder = nn.LSTM(100, 50, batch_first=True,
                               dropout=0.2, bidirectional=True)
        self.decoder = nn.LSTM(100, 50, batch_first=True,
                               dropout=0.2, bidirectional=True)
        self.enc_h0 = nn.Parameter(torch.empty(2, 50))
        self.enc_c0 = nn.Parameter(torch.empty(2, 50))
        self.dec_h0 = nn.Parameter(torch.empty(2, 50))
        self.dec_c0 = nn.Parameter(torch.empty(2, 50))
        nn.init.normal_(self.enc_h0)
        nn.init.normal_(self.enc_c0)
        nn.init.normal_(self.dec_h0)
        nn.init.normal_(self.dec_c0)

        self.bottleneck_down = nn.Linear(100, 25)
        self.bottleneck_up = nn.Linear(25, 100)

        self.relu = nn.ReLU()

    def infer(self, input_ids, attention_mask):
        embeds = self.embedding(input_ids)

        enc_h0 = torch.stack(input_ids.shape[0]*[self.enc_h0], dim=1)
        enc_c0 = torch.stack(input_ids.shape[0]*[self.enc_c0], dim=1)
        z, (enc_hn, enc_cn) = self.encoder(embeds, (enc_h0, enc_c0))

        z = self.bottleneck_down(z)
        z = self.bottleneck_up(z)

        dec_h0 = torch.stack(input_ids.shape[0]*[self.dec_h0], dim=1)
        dec_c0 = torch.stack(input_ids.shape[0]*[self.dec_c0], dim=1)
        x_hat, (dec_hn, dec_cn) = self.decoder(z, (dec_h0, dec_c0))

        logits = self.lm_head(x_hat)
        return logits

    def step(self, batch, batch_idx):
        input_ids, attention_mask = batch
        label_ids = input_ids*attention_mask + (attention_mask-1)*100

        logits = self.infer(input_ids, attention_mask)

        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            label_ids.view(-1),
            ignore_index=-100,
        )
        return loss, {"loss": loss}

    def score(self, batch, batch_idx):
        input_ids, attention_mask = batch
        label_ids = input_ids*attention_mask + (attention_mask-1)*100

        logits = self.infer(input_ids, attention_mask)

        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            label_ids.view(-1),
            ignore_index=-100,
            reduction="none",
        )
        score = loss.view(-1, logits.shape[1]).mean(-1)
        return score

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def predict_step(self, batch, batch_idx):
        loss = self.score(batch, batch_idx)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)



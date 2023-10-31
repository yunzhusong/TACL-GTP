import os
import json
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from torchtext.vocab import GloVe, vocab
from torchtext.transforms import VocabTransform
from torchtext.data import get_tokenizer
from torchtext.functional import to_tensor

from pytorch_lightning import LightningDataModule
from glob import glob

class Tokenizer:
    def __init__(self, base_tokenizer, vocab):
        self.base_tokenizer = base_tokenizer
        self.vocab_transform = VocabTransform(vocab)
        self.pad_token_id = vocab["<pad>"]

    def __call__(self, sents):
        return to_tensor(
            self.vocab_transform([self.base_tokenizer(sent) for sent in sents]),
            padding_value=self.pad_token_id,
        )

class UserDataset(Dataset):
    def __init__(self):
        # Load GloVe
        glove = GloVe(name="6B", dim=100, max_vectors=50000)

        # Vocabulary
        glove_tokens = {t:0 for t in glove.stoi.keys()} # give dummy occurance 0
        special_tokens = ["<unk>", "<s>", "</s>", "<pad>"]
        self.vocab = vocab(glove_tokens, min_freq=0, specials=special_tokens)
        self.vocab.set_default_index(self.vocab["<unk>"])

        # Tokenizer
        base_tokenizer = get_tokenizer("basic_english")
        self.tokenizer = Tokenizer(base_tokenizer, self.vocab)

    def setup(self, stage, *args, **kwargs):

        if stage == "fit":
            user_df = pd.read_pickle(os.path.join(kwargs["dataset_dir"], "test.pkl"))
            texts = sum([i.split(";;") for i in user_df["title"].tolist()], [])
 
        elif stage == "predict":

            if kwargs["predict_title_file"] == "user_title":
                user_df = pd.read_pickle(os.path.join(kwargs["dataset_dir"], "test.pkl"))
                texts = sum([i.split(";;") for i in user_df["title"].tolist()], [])

            elif kwargs["predict_title_file"] == "random200_editor_title":
                user_df = pd.read_pickle(os.path.join(kwargs["dataset_dir"], "test.pkl"))
                news_df = pd.read_pickle(os.path.join(kwargs["dataset_dir"], "news.pkl"))
                random_index = np.random.randint(0, len(news_df), 200*103)
                texts = news_df["title"].iloc[random_index].tolist()

            elif kwargs["predict_title_file"] == "editor_title":
                user_df = pd.read_pickle(os.path.join(kwargs["dataset_dir"], "test.pkl"))
                news_id = sum(user_df["posnewsID"].tolist(), [])
                news_df = pd.read_pickle(os.path.join(kwargs["dataset_dir"], "news.pkl"))
                texts = news_df["title"].iloc[news_id].tolist()

            else:
                with open(kwargs["predict_title_file"]+"/generated_predictions.txt", "r") as f:
                    texts = [s.strip() for s in f.readlines()]
                    #assert len(texts) == 20600
        else:
            raise ValueError("Invalid setup mode.")

        self.input_ids = self.tokenizer(texts)
        self.attention_mask = (self.input_ids!=self.vocab["<pad>"]).long()

    def setup_userwise(self, stage, *args, **kwargs):

        if stage == "fit":
            #test_df = pd.read_csv(os.path.join(kwargs["dataset_dir"], "test.csv"))
            test_df = pd.read_pickle(os.path.join(kwargs["dataset_dir"], "test.pkl"))
            user_df = test_df[test_df["userID"]==kwargs["user_id"]]
            texts = user_df["title"].item().split(";;")
        elif stage == "predict":

            with open(kwargs["predict_title_file"], "r") as f:
                texts = [s.strip() for s in f.readlines()]
        else:
            raise ValueError("Invalid setup mode.")

        self.input_ids = self.tokenizer(texts)
        self.attention_mask = (self.input_ids!=self.vocab["<pad>"]).long()

    def setup_multiple_prediction(self, stage, *args, **kwargs):

        split_df = pd.read_csv(kwargs["predict_title_file"]+"/split.csv")
        self.care_data_mask = []
        care_data_mask =  (split_df["split"]=="test").to_list()

        texts = []
        for i in range(103):
            with open(kwargs["predict_title_file"]+f"/NT{i+1}/generated_predictions.txt", "r") as f:
                titles = [s.strip() for s in f.read().split(";;")]

            df = pd.DataFrame({"titles": [""]*200})
            df.loc[care_data_mask, "titles"] = titles
            texts += df["titles"].tolist()

        self.input_ids = self.tokenizer(texts)
        self.attention_mask = (self.input_ids!=self.vocab["<pad>"]).long()
        self.care_data_mask = care_data_mask*103

    def setup_prediction_extract_interuser(self, stage, *args, **kwargs):

        data_path = kwargs["predict_title_file"]
        better_checkpoint = self.find_better_checkpoint(data_path)
        with open(f"{better_checkpoint}/generated_predictions.txt", "r") as f:
            titles = [s.strip() for s in f.readlines()]

        split_df = pd.read_csv(data_path+"/split.csv")
        masks = (split_df["split"]=="test").tolist()

        title_df = pd.DataFrame({"titles": [" "]*20600})
        title_df.loc[masks,"titles"] = titles
        texts = title_df["titles"].tolist()

        self.input_ids = self.tokenizer(texts)
        self.attention_mask = (self.input_ids!=self.vocab["<pad>"]).long()
        self.care_data_mask = masks

    def setup_multiple_prediction_extract(self, stage, *args, **kwargs):

        #split_df = pd.read_csv(kwargs["predict_title_file"]+"/split.csv")
        #care_data_mask =  (split_df["split"]=="test").to_list()

        texts, masks = [], []
        for i in range(103):
            data_path = kwargs["predict_title_file"]+f"/index{i}"
            better_checkpoint = self.find_better_checkpoint(data_path)
            #with open(f"{data_path}/{better_checkpoint}/generated_predictions.txt", "r") as f:
            with open(f"{better_checkpoint}/generated_predictions.txt", "r") as f:
                titles = [s.strip() for s in f.readlines()]

            split_df = pd.read_csv(data_path+"/split.csv")
            mask = (split_df["split"]=="test").tolist()
 
            title_df = pd.DataFrame({"titles": [" "]*200})
            title_df.loc[mask,"titles"] = titles
            texts += title_df["titles"].tolist()
            masks += mask

        self.input_ids = self.tokenizer(texts)
        self.attention_mask = (self.input_ids!=self.vocab["<pad>"]).long()
        self.care_data_mask = masks

    def find_better_checkpoint(self, data_path):
        
        if "checkpoint" in data_path.split("/")[-1]:
            return data_path
        max_score = 0
        for checkpoint in glob(f"{data_path}/checkpoint*"):
            result_path = os.path.join(checkpoint, "all_results.json")
            #result_path = os.path.join(data_path, checkpoint, "all_results.json")

            f = open(result_path)
            results = json.load(f)
            score = results["predict_combined_score"]
            if score > max_score:
                best_checkpoint = checkpoint

        return best_checkpoint

    def __getitem__(self, index):
        return self.input_ids[index], self.attention_mask[index]

    def __len__(self):
        return len(self.input_ids)

class UserDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        user_id: str,
        fold_size: int,
        fold_id: int,
        predict_title_file: str,
        predict_mask_file: str,
        train_batch_size: int,
        eval_batch_size: int,
        num_workers: int,
        multiple_files: bool,
        file_extracted: bool,
        interuser: bool,
        **kwargs,
    ):
        super().__init__()

        # For training/validation
        self.dataset_dir = dataset_dir
        self.user_id = user_id
        self.fold_size = fold_size
        self.fold_id = fold_id

        # For prediction
        self.predict_title_file = predict_title_file
        self.predict_mask_file = predict_mask_file
        self.multiple_files = multiple_files
        self.extracted = file_extracted
        self.interuser = interuser

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers

        if user_id != "all":
            self.setup = self.setup_userwise

    def setup(self, stage: str):
        dataset = UserDataset()
        if stage == "fit":
            dataset.setup(stage, dataset_dir=self.dataset_dir, user_id=self.user_id)
            assert len(dataset) % self.fold_size == 0
            assert self.fold_id < int(len(dataset) / self.fold_size)
        elif stage == "predict":
            if self.interuser:
                dataset.setup_prediction_extract_interuser(stage, predict_title_file=self.predict_title_file, 
                              dataset_dir=self.dataset_dir)
            elif self.multiple_files and self.extracted:
                dataset.setup_multiple_prediction(stage, predict_title_file=self.predict_title_file, 
                              dataset_dir=self.dataset_dir)
            elif self.multiple_files:
                dataset.setup_multiple_prediction_extract(stage, predict_title_file=self.predict_title_file, 
                              dataset_dir=self.dataset_dir)
            else:
                dataset.setup(stage, predict_title_file=self.predict_title_file, 
                              dataset_dir=self.dataset_dir)
        else:
            raise ValueError("Invalid dataset mode.")

        #assert len(dataset) % self.fold_size == 0
        #assert self.fold_id < int(len(dataset) / self.fold_size)

        all_idx = np.arange(len(dataset)).reshape(-1, 200)
        eval_idx = all_idx[:,self.fold_id*self.fold_size:(self.fold_id+1)*self.fold_size].reshape(-1)
        train_idx = list(set(all_idx.reshape(-1)) - set(eval_idx))

        # NOTE: Predict on train or eval data
        self.train_dataset = Subset(dataset, train_idx)
        self.eval_dataset = Subset(dataset, eval_idx)

        #self.train_dataset = Subset(dataset, eval_idx)
        #self.eval_dataset = Subset(dataset, train_idx)

        if stage == "predict":
            if self.predict_mask_file:
                with open(self.predict_mask_file, "r") as f:
                    care_data_mask = [bool(int(m)) for m in f.read().strip().split(",")]
                self.train_care_data_mask = [care_data_mask[i] for i in train_idx]
                self.eval_care_data_mask = [care_data_mask[i] for i in eval_idx]

            elif self.multiple_files or self.interuser:
                care_data_mask = dataset.care_data_mask
                self.train_care_data_mask = [care_data_mask[i] for i in train_idx]
                self.eval_care_data_mask = [care_data_mask[i] for i in eval_idx]

            else:
                self.train_care_data_mask = [True]*len(train_idx)
                self.eval_care_data_mask = [True]*len(eval_idx)

    def setup_userwise(self, stage: str):
        dataset = UserDataset()
        if stage == "fit":
            dataset.setup_userwise(stage, dataset_dir=self.dataset_dir, user_id=self.user_id)
        elif stage == "predict":
            dataset.setup_userwise(stage, predict_title_file=self.predict_title_file)
        else:
            raise ValueError("Invalid dataset mode.")

        assert len(dataset) % self.fold_size == 0
        assert self.fold_id < int(len(dataset) / self.fold_size)
        eval_idx = list(range(self.fold_id*self.fold_size, (self.fold_id+1)*self.fold_size))
        train_idx = list(set(range(len(dataset)))-set(eval_idx))

        self.train_dataset = Subset(dataset, eval_idx)
        self.eval_dataset = Subset(dataset, train_idx)

        #self.train_dataset = Subset(dataset, train_idx)
        #self.eval_dataset = Subset(dataset, eval_idx)

        if stage == "predict":
            with open(self.predict_mask_file, "r") as f:
                care_data_mask = [int(m) for m in f.read().strip().split(",")]
            self.train_care_data_mask = [care_data_mask[i] for i in train_idx]
            self.eval_care_data_mask = [care_data_mask[i] for i in eval_idx]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.train_batch_size,
                          num_workers=self.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.eval_dataset,
                          batch_size=self.eval_batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.eval_dataset,
                          batch_size=self.eval_batch_size,
                          num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.eval_dataset,
                          batch_size=self.eval_batch_size,
                          num_workers=self.num_workers)

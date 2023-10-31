""" Anomaly-based Personalization Evaluation (APE). """
import os
import numpy as np

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from ape.config import ex
from ape.data import UserDataModule
from ape.models import LSTMBottleAutoEncoder, TransformerBottleAutoEncoder

@ex.automain
def main(_config):

    seed_everything(_config["seed"])

    dm = UserDataModule(
        dataset_dir=_config["dataset_dir"],
        user_id=_config["user_id"],
        fold_size=_config["fold_size"],
        fold_id=_config["fold_id"],
        predict_title_file=_config["predict_title_file"],
        predict_mask_file=_config["predict_mask_file"],
        train_batch_size=_config["train_batch_size"],
        eval_batch_size=_config["eval_batch_size"],
        num_workers=_config["num_workers"],
        multiple_files=_config["multiple_files"],
        file_extracted=_config["file_extracted"],
        interuser=_config["interuser"],
    )

    if _config["model_type"] == "lstm_bottleneck":
        model_class = LSTMBottleAutoEncoder
    elif _config["model_type"] == "transformer_bottleneck":
        model_class = TransformerBottleAutoEncoder
    else:
        raise ValueError("Invalid model type.")

    if _config["checkpoint_path"] is not None:
        model = model_class.load_from_checkpoint(_config["checkpoint_path"])
    else:
        model = model_class()

    save_dir = "../results/ape/" + ('predict' if _config['predict_title_file'] is not None else 'train')
    name = f"{_config['exp_name']}/{_config['user_id']}/fold_{_config['fold_id']}"
    tb_logger = TensorBoardLogger(
        save_dir=save_dir,
        name=name,
        default_hp_metric=False,
        flush_secs=3,
    )

    checkpoint_callback = ModelCheckpoint(
        filename="epoch_{epoch}_step_{step}",
        auto_insert_metric_name=False,
        monitor=_config["monitor"],
        mode=_config["metric_mode"],
        every_n_epochs=1,
        save_top_k=_config["save_top_k"],
    )
    early_stop_callback = EarlyStopping(
        monitor=_config["monitor"],
        mode=_config["metric_mode"],
        min_delta=0.0,
        patience=_config["patience"],
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(
        accelerator="gpu",
        devices=[i for i in range(torch.cuda.device_count())],
        deterministic=True,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=tb_logger,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        gradient_clip_val=1.0,
        max_epochs=_config["max_epochs"],
    )

    if _config["predict_title_file"] is not None and _config["fad"]:
        dm.setup("predict")
        features = []
        with torch.no_grad():
            for input_ids, attention_mask in dm.predict_dataloader():
                z = model.bottleneck(input_ids, attention_mask)
                #z = model.decoder(model.bottleneck(input_ids, attention_mask))
                # Not calculate the padding token
                #mask = attention_mask.unsqueeze(-1).repeat(1,1,z.shape[-1])
                #mask[:,0,:] = 1
                # Remove the empty data
                #not_empty = torch.sum(attention_mask, dim=1)!=0
                #features.append(np.average(z[not_empty], axis=1, weights=mask[not_empty]))
                #features.append(torch.sum(z*mask, dim=1)/torch.sum(mask, dim=1))
                #features.append(np.average(z, axis=1, weights=mask))
                #features.append(np.array(torch.sum(z*mask, dim=1) / torch.sum(mask, dim=1)))
                #features.append(np.array(torch.sum(z*mask, dim=1)))
                features.append(np.average(z, axis=1))
            features = np.concatenate(features, axis=0)
            #features = np.array(torch.concat(features, dim=0)[dm.eval_care_data_mask])

        mu = np.mean(features[dm.eval_care_data_mask], axis=0)
        sigma = np.cov(features[dm.eval_care_data_mask], rowvar=False)
 
        output_dir = os.path.join(save_dir, name)
        os.makedirs(output_dir, exist_ok=True)
        np.savez(os.path.join(output_dir, "statistic_bottleneck.npz"),
                mu=mu, sigma=sigma, features=features, eval_care_data_mask=dm.eval_care_data_mask)

    elif _config["predict_title_file"] is not None:
        results = trainer.predict(model, datamodule=dm)
        results = torch.concat(results)

        with open(os.path.join(trainer.logger.log_dir, "anomaly_scores.txt"), "w") as f:
            for r in results:
                f.write(f"{r}\n")
        with open(os.path.join(trainer.logger.log_dir, "score_masks.txt"), "w") as f:
            f.write(",".join([str(m) for m in dm.eval_care_data_mask]))
    else:
        trainer.fit(model, datamodule=dm)

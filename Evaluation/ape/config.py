from sacred import Experiment

ex = Experiment("APE")

@ex.config
def config():
    exp_name = "ape"
    dataset_dir="../datasets/pens/"
    model_type = "transformer_bottleneck"

    # Experimental setting
    interuser = False

    # User setting
    user_id = "NT1"
    fold_size = 40
    fold_id = 0

    # Optimizer Setting
    max_epochs = 1000

    # PL Setting
    seed = 0
    monitor = "val_loss"
    metric_mode = "min"
    save_top_k = 1
    patience = 5

    # Varies
    train_batch_size = 64
    eval_batch_size = 64
    num_workers = 8
    checkpoint_path = None

    # Predict Setting
    predict_title_file = None
    predict_mask_file = None

    # Frechet distance
    fad = True

    # For encoding different files
    multiple_files = False
    file_extracted = False

@ex.named_config
def transformer():
    exp_name = "transformer"
    model_type = "transformer"

@ex.named_config
def lstm():
    exp_name = "lstm"
    model_type = "lstm"

@ex.named_config
def transformer_bottleneck():
    exp_name = "transformer_bottleneck"
    model_type = "transformer_bottleneck"

@ex.named_config
def lstm_bottleneck():
    exp_name = "lstm_bottleneck"
    model_type = "lstm_bottleneck"


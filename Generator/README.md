
## 1. Create dataset from raw dataset by expanding the positive samples.
- Convert from recommendation dataset to headline generation dataset
- There are inter-user and intra-user setups
```
python ./preprocess/make_dataset.py
```

## 2. Create dataset for few-shot finetuning.
- Default setup is 5 fold and 5-shot.
- Save in *../datasets/specialize_own/intra_5user/[0-4]*
```
python ./preprocess/make_finetuning_dataset.py
```

## 3. Model Pretraining

**Pretrain Objectives**
- Text infill (follow BART)
- Text shuffle (token-level)
*TODO: build span-level shuffle*
- Mask User Modeling: recover masked user token embedding, mask the token with a new create token embedding

**Input Setup**
- userize: whether to append the user embedding
- extra: whether to append the extracted sentence 
```
./scripts/pretrain.sh
```

## 4. Few-shot Finetuning
*inter-user few-shot*
- Finetune the model with a few **test samples** (5/103 for train and 5/103 for validation), and test on the rest samples (93/103)
- Set *train_file* and *validation_file*
- Control by setting *do_finetine=True*
```
./scripts/finetune.sh
```
*user-wise few-shot*
```
./scripts/userwise.sh
```

## 5. Evaluate model performance
- Make sure we specify the *test_file* with correct fold
- Take the 1st stage predictions as input text (specify *predict_txt_file*)
```
python main.py scripts/args/predict.json
```


---
### Argument Explaination
- Input Control:
```--text_file``` *Default: "../results/pens_ghg_own/bart/checkpoint-98000/all_news/generated_predictions.txt", use the text_file to specify the input text, use the ```--text_column``` to get input text from news_file if None (may need corruption for better training).*
```--text_column``` To specify the input column.
```--extra_info``` Whether append the retrieved news body after input text
```--retrieve_online```


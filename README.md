# Dacon_KNLI

## About this project
```
This project launched in Dacon, Korean data scientists' community, where similar to kaggle.
Purpose of participation of the project is to improve myself.
The project's subject is kind of basic NLI(Natural Language Inference).
I used roBERTa as base, and trained with Dacon data and kakao_knli_dataset.
```

## Used
### Model
```
klue/roberta-large
```
### Data
```
Dacon Data
kakao_knli_dataset
```

## Directory Tree
```bash
📦Dacon_knli
 ┣ 📂configs
 ┃ ┣ 📜cfg.yaml
 ┃ ┣ 📜setting.py
 ┃ ┣ 📜template.yaml
 ┃ ┗ 📜__init__.py
 ┣ 📂dataload
 ┃ ┣ 📜dataloader.py
 ┃ ┣ 📜dataset.py
 ┃ ┣ 📜lit_dataloader.py
 ┃ ┗ 📜__init__.py
 ┣ 📂model
 ┃ ┣ 📜lit_model.py
 ┃ ┣ 📜loss_load.py
 ┃ ┣ 📜model_load.py
 ┃ ┣ 📜opt_load.py
 ┃ ┗ 📜__init__.py
 ┣ 📂utils
 ┃ ┣ 📂callbacks
 ┃ ┃ ┣ 📜early_stopping.py
 ┃ ┃ ┣ 📜save_checkpoint.py
 ┃ ┃ ┗ 📜__init__.py
 ┃ ┣ 📂common
 ┃ ┃ ┣ 📜common.py
 ┃ ┃ ┣ 📜project_paths.py
 ┃ ┃ ┗ 📜__init__.py
 ┃ ┣ 📂core
 ┃ ┃ ┣ 📜trainer.py
 ┃ ┃ ┗ 📜__init__.py
 ┃ ┣ 📂logger
 ┃ ┃ ┣ 📜csv_logger.py
 ┃ ┃ ┗ 📜__init__.py
 ┃ ┣ 📜train_utils.py
 ┃ ┗ 📜__init__.py
 ┣ 📜.gitignore
 ┣ 📜inference.py
 ┣ 📜lit_main.py
 ┣ 📜main.py
 ┣ 📜README.md
 ┗ 📜requirements.txt
```

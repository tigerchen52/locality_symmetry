# Source Code
This is a Pytorch implementation of BERT code, adapted from [PytorchicBERT](https://github.com/dhlee347/pytorchic-bert). 
```
└───source_code
    │   checkpoint.py
    │   check_step.py
    │   classify.py
    │   debug.py
    │   glue_eval.py
    │   optim.py
    │   pretrain.py
    │   README.md
    │   requirements.txt
    │   tokenization.py
    │   train.py
    │   utils.py
    │   __init__.py
    │
    ├───config
    │       bert_base.json
    │       pretrain.json
    │       train_mrpc.json
    │
    ├───data
    │   └───snli
    │           random_shuffle.txt
    │           shuffle_3.txt
    │           shuffle_4.txt
    │           shuffle_5.txt
    │           shuffle_6.txt
    │           shuffle_sr.txt
    │
    └───model
            attenuated.py
            attenuated_l.py
            bert.py
            bert_a_l_s.py
            concat.py
            seuqence.py
            __init__.py
```


## Environment setup
Clone the repository and set up the environment via "requirements.txt" and here we use python3.
```
pip install -r requirements.txt
```

## Data preparation
We use Wikipedia to pre-train our models, you can download related dataset from this [web page](https://dumps.wikimedia.org/backup-index.html).
The version is 20200101 dumps.
As for preprocessing, we adopt [WikiExtractor](https://github.com/attardi/wikiextractor), which is a Python script that extracts and cleans text from a Wikipedia database dump.

## Training and Evaluation
All hyper-parameters of our model are stored in `configs/`, and we can adjust these values to obtain the corresponding models in our paper.<br>
### Pre-training 
Input file format :
1. One sentence per line. These should ideally be actual sentences, not entire paragraphs or arbitrary spans of text. (Because we use the sentence boundaries for the "next sentence prediction" task).
2. Blank lines between documents. Document boundaries are needed so that the "next sentence prediction" task doesn't span between documents.
```
Document 1 sentence 1
Document 1 sentence 2
...
Document 1 sentence 45

Document 2 sentence 1
Document 2 sentence 2
...
Document 2 sentence 24
```
Usage :
```
export DATA_FILE=/path/to/corpus
export BERT_PRETRAIN=/path/to/pretrain
export SAVE_DIR=/path/to/save

python pretrain.py \
    --train_cfg config/pretrain.json \
    --model_cfg config/bert_base.json \
    --data_file $DATA_FILE \
    --vocab $BERT_PRETRAIN/vocab.txt \
    --save_dir $SAVE_DIR \
    --max_len 512 \
    --max_pred 20 \
    --mask_prob 0.15
```

### Evaluation
We evaluate on 10 subsets of [SentEval](https://github.com/facebookresearch/SentEval) and [GLUE (General Language Understanding)](https://gluebenchmark.com/) benchmark, which is a collection of language understanding
tasks including sentiment analysis, textual entailment and so on.

Usage :
 ```
python glue_eval.py
```


## PROCESSING SPECS

- question for shots only
- images are resized to 224X224 for preprocessing (resnet input size)
- using last layer of resnet50 for feature extraction
- using glove.6B.300d for pretrained word embedding
- storing image feature cache after feature extraction (for faster dataloading)
- using random splits for train, test, val (8: 1: 1) respectively
- using multiprocessing for faster processing
- using nltk.word_tokenize for tokenization
- You should first run **json_to_jsonl.py** before running the preprocessing code
  to change the question data formats (this procedure may be integrated to the main code upon discussion)

The data folder should be structured as follows:

> ./data
> ./data/QA_train_set_s1s2.json
> ./data/images/s0101/...

The preprocessing command should be the following:

> python cli.py check_dataloader

for now, at least.

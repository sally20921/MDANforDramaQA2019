# Multimodal dual attention networks for 2019 DramaQA challenge 

This file contains code to conduct Drama QA with Multimodal dual attention networks.
data/ (preprocessing) 
data loader, supports image loading, feature extraction, feature caching
model/ (attention module, multimodal fusion)
-'attention_fusion.py' : code for Multimodal dual attention networks model
-'temporal_graph.py': submodules 

## What has changed from starter code (https://github.com/skaro94/vtt_challenge_2019)

A new model has been added (Multimodal dual attention networks)
Specified in 'attention_fusion.py'
Files needed to train the model has been changed accordingly
('config.py', 'train.py', 'ckpt.py' etc.)
 
## Dependency

We use python3 (3.5.2), and python2 is not supported.
We use PyTorch (1.1.0), though tensorflow-gpu is necessary to launch tensorboard.

python packages:
[fire](https://github.com/google/python-fire) for commandline api


## Data Folder Structure

```
data/
  AnotherMissOh/
    AnotherMissOh_images/
      $IMAGE_FOLDERS
    AnotherMissOh_QA/
      AnotherMissOhQA_train_set.json
      AnotherMissOhQA_val_set.json
      AnotherMissOhQA_test_set.json
      $QA_FILES
    AnotherMissOh_subtitles.json
```

## Install

```bash
git clone --recurse-submodules (this repo)
cd $REPO_NAME/code
(use python >= 3.5)
pip install -r requirements.txt
python -m nltk.downloader 'punkt'
```

Place the data folder at `data`.

## How to Use

### training

```bash
cd code
python cli.py train
```

Access the prompted tensorboard port to view basic statistics.
At the end of every epoch, a checkpoint file will be saved on `/data/ckpt/OPTION_NAMES`

- Use `video_type` config option to use `'shot'` or `'scene'` type data.

- if you want to run the code with less memory requirements, use the following flags.
```bash
python cli.py train --extractor_batch_size=$BATCH --num_workers=$NUM_WORKERS
```

- You can use `use_inputs` config option to change the set of inputs to use.
The default value is `['images', 'subtitle']`.
It is forbidden to use `description` input for the challenge.

For further configurations, take a look at `startup/config.py` and
[fire](https://github.com/google/python-fire).

### evaluation

```bash
cd code
python cli.py evaluate --ckpt_name=$CKPT_NAME
```

Substitute CKPT_NAME to your prefered checkpoint file.
e\.g\. `--ckpt_name=='feature*/loss_1.34'`

### making submissions

```bash
python cli.py infer --model_name=$MODEL_NAME --ckpt_name=$CKPT_NAME
```

The above command will save the outcome at the prompted location.

### evaluating submissions

```bash
cd code/scripts
python eval_submission.py -y $SUBMISSION_PATH -g $DATA_PATH
```

### Default Preprocessing Details

- images are resized to 224X224 for preprocessing (resnet input size)
- using last layer of resnet50 for feature extraction (base behaviour)
- using glove.6B.300d for pretrained word embedding
- storing image feature cache after feature extraction (for faster dataloading)
- using nltk.word_tokenize for tokenization
- all images for a scene questions are concatenated in a temporal order

## 

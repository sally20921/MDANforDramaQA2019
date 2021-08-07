# Documentation

## files

- `cli.py`: commandline interface
- `config.py`: default configs
- `ckpt.py`: checkpoint saving & loading
- `train.py`: trainer
- `evaluate.py`: evaluate accuracy with target
- `infer.py`: make submission from checkpoint
- `interactive.py`: interactive inference
- `logger.py`: tensorboard and commandline logger for scalars
- `utils.py`: utils
- `json_to_jsonl.py`: json -> line json
- `loss`: loss
  - `cross_entropy.py`: cross entropy loss
- `metric`: metric (accuracy, ngram)
  - `ngram.py`: ngram metric for generation
  - `stat_metric.py`: accuracy and loss logging
- `optimizer`: AdaGrad optimizer
- `scripts`: standalone scripts
  - `eval_submission.py`: evaluate submission file
- `data`: dataloader
  - `dataset`: dataloader for generation
  - `dataset_multichoice`: dataloader for classification
  - `preprocess_image.py`: image feature extraction
  - `vision.py`: image loading helper

# functions

- `train.train`: training
- `evaluate.evaluate_once`: evaluate for single epoch
- `dataset_multichoice.get_iterator`: load data iterator
- `preprocess_image.preprocess_images`: extract all image features
- `infer.infer`: make submission file
- `utils.prepare_batch`: move to GPU and build target
- `ckpt.get_model_ckpt`:  load ckpt and substitute model weight, vocab, and args

config = {
    'multichoice': True,
    'extractor_batch_size': 384,
    'model_name': 'multi_choice',
    'log_path': 'data/log',
    'tokenizer': 'nltk',
    'batch_sizes': (12, 12, 12),
    'lower': True,
    'use_inputs': ['images', 'subtitle'],  # We advise not to use description for the challenge
    'cache_image_vectors': True,
    'image_path': 'data/AnotherMissOh/AnotherMissOh_images',
    'data_path': 'data/AnotherMissOh/AnotherMissOh_QA/AnotherMissOhQA_set_subtitle.jsonl',
    'subtitle_path': 'data/AnotherMissOh/AnotherMissOh_subtitles.json',
    'vocab_pretrained': "glove.6B.300d",
    'video_type': ['shot', 'scene'],
    'feature_pooling_method': 'mean',
    'max_epochs': 20,
    'allow_empty_images': False,
    'num_workers': 40,
    'image_dim': 512,  # hardcoded for ResNet50
    'n_dim': 256,
    'layers': 3,
    'dropout': 0.5,
    'learning_rate': 0.001,
    'loss_name': 'cross_entropy_loss',
    'optimizer': 'adagrad',
    # 'metrics': ['bleu', 'rouge'],
    'metrics': [],
    'log_cmd': False,
    'ckpt_path': 'data/ckpt',
    'ckpt_name': None,
    'text_feature_names': ['subtitle', 'description'],
}


debug_options = {
    # 'image_path': './data/images/samples',
}

log_keys = [
    'model_name',
    'feature_pooling_method',
]

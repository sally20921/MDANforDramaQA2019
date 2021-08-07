from functools import partial
from collections import defaultdict

import torch
from torchtext import data
import nltk

from utils import pad_tensor, make_jsonl
from .load_subtitle import update_subtitle
from .preprocess_image import preprocess_images, get_empty_image_vector


def load_text_data(args, tokenizer, vocab=None):
    vid = InfoField()
    qid = InfoField()
    videoType = InfoField()
    q_level_mem = InfoField()
    q_level_logic = InfoField()
    correct_idx = InfoField()

    que = data.Field(sequential=True, tokenize=tokenizer, lower=args.lower)
    description = data.Field(sequential=True, tokenize=tokenizer, lower=args.lower)
    subtitle = data.Field(sequential=True, tokenize=tokenizer, lower=args.lower)
    single_answer = data.Field(sequential=True, tokenize=tokenizer, lower=args.lower,
                          init_token='<sos>', eos_token='<eos>',
                          unk_token='<unk>')
    answers = data.NestedField(single_answer)

    common_fields = {
        'vid': ('vid', vid),
        'qid': ('qid', qid),
        'videoType': ('videoType', videoType),
        'q_level_mem': ('q_level_mem', q_level_mem),
        'q_level_logic': ('q_level_logic', q_level_logic),
        'que': ('que', que),
        'description': ('description', description),
        'subtitle': ('subtitle', subtitle),
        'answers': ('answers', answers),
    }

    non_test_fields = {
        'correct_idx': ('correct_idx', correct_idx),
    }

    text_data = []
    for mode in ['train', 'test', 'val']:
        name = args.data_path.name
        name = name.split('_')
        name.insert(1, mode)
        name = '_'.join(name)
        data_path = args.data_path.parent / name
        update_subtitle(data_path, args.subtitle_path)
        make_jsonl(data_path)

        fields = {**non_test_fields, **common_fields} if mode != 'test' \
            else common_fields

        text_data.append(data.TabularDataset(
            path=str(data_path), format='json',
            fields=fields))

    print("using {} videoType".format(args.video_type))
    for t in text_data:
        remove_questions(t, args.video_type)

    train_iter, test_iter, val_iter = data.Iterator.splits(
        tuple(text_data), sort_key=lambda x: len(x.que),
        batch_sizes=args.batch_sizes, device=args.device,
        sort_within_batch=True,
    )
    train = text_data[0]

    if vocab is None:
        vocab_args = {}
        k = 'vocab_pretrained'
        if hasattr(args, k):
            vocab_args['vectors'] = getattr(args, k)
        que.build_vocab(train.que,
                        train.answers, train.single_answer,
                        train.description, train.subtitle,
                        **vocab_args)
        que.vocab = process_vocab(que.vocab)
        vocab = que.vocab
    answers.vocab = vocab
    single_answer.vocab = vocab
    que.vocab = vocab
    description.vocab = vocab
    subtitle.vocab = vocab

    return {'train': train_iter, 'val': val_iter, 'test': test_iter}, vocab


def process_vocab(vocab):
    vocab.specials = ['<sos>', '<eos>', '<unk>', '<pad>']

    vocab.special_ids = [vocab.stoi[k] for k in vocab.specials]
    for token in vocab.specials:
        setattr(vocab, token[1:-1], token)

    return vocab


def get_tokenizer(args):
    return {
        'nltk': nltk.word_tokenize
    }[args.tokenizer.lower()]


class ImageIterator:
    def __init__(self, args, text_it):
        self.it = text_it
        self.args = args
        self.allow_empty_images = args.allow_empty_images
        self.num_workers = args.num_workers
        self.device = args.device

        self.image_dt = self.load_images(args.image_path, text_it.dataset,
                                         cache=args.cache_image_vectors, device=args.device)
        print("total vids: {}".format(len(list(self.image_dt))))

    def __iter__(self):
        for batch in self.it:
            batch.images = self.get_image(batch.vid)
            yield batch

    def get_image(self, vids):
        images = [torch.from_numpy(self.image_dt[vid]).to(self.device).split(1) for vid in vids]
        images = pad_tensor(images).squeeze(2)

        return images

    def load_images(self, image_path, dataset, cache=True, device=-1):
        images = preprocess_images(self.args, image_path, cache=cache, device=device, num_workers=self.num_workers)
        if self.allow_empty_images:
            for k, v in images.items():
                sample_image = v
                break
            func = partial(get_empty_image_vector, sample_image_size=list(sample_image.shape))
            images = defaultdict(func, images)

        return {ex.vid: images[ex.vid] for ex in dataset}


def get_image_iterator(args, text_it):
    return ImageIterator(args, text_it)


# batch: [len, batch_size]
def get_iterator(args, vocab=None):
    print("Loading Text Data")
    tokenizer = get_tokenizer(args)
    iters, vocab = load_text_data(args, tokenizer, vocab)
    print("Loading Image Data")
    image_iters = {}
    for key, it in iters.items():
        image_iters[key] = get_image_iterator(args, it)
    print("Data Loading Done")

    return image_iters, vocab


def remove_questions(dataset, video_types):
    li = []
    for example in dataset.examples:
        if example.videoType in video_types:
            li.append(example)

    dataset.examples = li


class InfoField(data.RawField):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.is_target = False

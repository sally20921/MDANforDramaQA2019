from contextlib import contextmanager
from datetime import datetime

import os
import sys
import json

import six
import numpy as np
import torch

from config import log_keys


def get_dirname_from_args(args):
    dirname = ''
    for key in sorted(log_keys):
        dirname += '_'
        dirname += key
        dirname += '_'
        dirname += str(args[key])

    return dirname[1:]


def get_now():
    now = datetime.now()
    return now.strftime('%Y-%m-%d-%H-%M-%S')


def make_jsonl(path):
    if not path.is_file() and path.suffix == '.jsonl':
        path_json = path.parent / path.name[:-1]
        with open(str(path_json), 'r') as f:
            li = json.load(f)
        with open(str(path), 'w') as f:
            for line in li:
                f.write("{}\n".format(json.dumps(line)))


def prepare_batch(args, batch, vocab):
    # transport device
    for name in dir(batch):
        val = getattr(batch, name)
        if torch.is_tensor(val):
            setattr(batch, name, val.to(args.device).contiguous())

    # target = getattr(batch, tgt_key)[:,1:]  # ditch sos
    # TODO: ditch eos
    # shifted_target = getattr(batch, tgt_key)[:,:-1]  # ditch eos

    batch.que = batch.que.transpose(0, 1)
    batch.description = batch.description.transpose(0, 1)
    batch.subtitle = batch.subtitle.transpose(0, 1)
    if not hasattr(batch, 'answers'):
        # merge answers
        false_ans = batch.false_ans  # B4L
        true_ans = batch.true_ans.transpose(0, 1).unsqueeze(1)  # B1L'
        max_len = max(false_ans.shape[-1], true_ans.shape[-1])
        # pad
        padding = torch.Tensor([vocab.stoi[vocab.pad]]).long().to(true_ans.device)
        false_ans = torch.cat([false_ans, padding.view(1, 1, 1).expand(*false_ans.shape[:2], max_len - false_ans.shape[-1])],
                            dim=-1)
        true_ans = torch.cat([true_ans, padding.view(1, 1, 1).expand(*true_ans.shape[:2], max_len - true_ans.shape[-1])],
                            dim=-1)
        B = true_ans.shape[0]
        A = false_ans.shape[1] + true_ans.shape[1]
        L = true_ans.shape[-1]
        # random join
        ans_idx = torch.randint(0, A, (B,), dtype=torch.long).to(true_ans.device)
        answers = torch.zeros((B, A, L)).long().to(true_ans.device)
        answers.scatter_(1, ans_idx.view(-1, 1, 1).expand(-1, 1, answers.shape[-1]),
                        true_ans.expand(-1, answers.shape[1], -1))
        uidx = torch.arange(0, A).view(1, -1).expand(ans_idx.shape[0], -1).to(true_ans.device)
        k = (uidx != ans_idx.unsqueeze(-1)).nonzero()
        uidx = uidx[k[:, 0], k[:, 1]].view(-1, false_ans.shape[1])
        answers.scatter_(1, uidx.unsqueeze(-1).expand(-1, -1, L), false_ans)
        batch.answers = answers
    else:
        if hasattr(batch, 'correct_idx'):
            ans_idx = torch.Tensor(batch.correct_idx).to(batch.answers.device).long()
            # ans_idx = ans_idx - 1  # 1~5 -> 0~4
        else:
            ans_idx = None
    '''
        # pad answer
        answers = batch.answers
        padding = torch.Tensor([vocab.stoi[vocab.pad]]).long().to(answers[0].device)
        answers = torch.cat([answers, padding.view(1, 1, 1).expand(*answers.shape[:2], max_len - answers.shape[-1])],
                            dim=-1)
        batch.answers = answers
    '''

    net_input_key = ['que', 'answers', *args.use_inputs]
    net_input = {k: getattr(batch, k) for k in net_input_key}

    return net_input, ans_idx


def to_string(vocab, x):
    # x: float tensor of size BSZ X LEN X VOCAB_SIZE
    # or idx tensor of size BSZ X LEN
    if x.dim() > 2:
        x = x.argmax(dim=-1)

    res = []
    for i in range(x.shape[0]):
        sent = x[i]
        li = []
        for j in sent:
            if j not in vocab.special_ids:
                li.append(vocab.itos[j])
        sent = ' '.join(li)
        res.append(sent)

    return res


def wait_for_key(key="y"):
    text = ""
    while (text != key):
        text = six.moves.input("Press {} to quit".format(key))
        if text == key:
            print("terminating process")
        else:
            print("key {} unrecognizable".format(key))


def get_max_size(t):
    if hasattr(t, 'shape'):
        if not torch.is_tensor(t):
            t = torch.from_numpy(t)
        return list(t.shape), t.dtype
    else:
        # get max
        t = [get_max_size(i) for i in t]
        dtype = t[0][1]
        t = [i[0] for i in t]
        return [len(t), *list(np.array(t).max(axis=0))], dtype


def pad_tensor(x, val=0):
    max_size, dtype = get_max_size(x)
    storage = torch.full(max_size, val).type(dtype)

    def add_data(ids, t):
        if hasattr(t, 'shape'):
            if not torch.is_tensor(t):
                t = torch.from_numpy(t)
            storage[tuple(ids)] = t
        else:
            for i in range(len(t)):
                add_data([*ids, i], t[i])

    add_data([], x)

    return storage


@contextmanager
def suppress_stdout(do=True):
    if do:
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout

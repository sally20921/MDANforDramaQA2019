from pathlib import Path
import random

from fire import Fire
from munch import Munch

import torch
import numpy as np

from config import config, debug_options
from data.dataset import get_iterator
from utils import wait_for_key, suppress_stdout
from train import train
from evaluate import evaluate
from interactive import interactive
from infer import infer


class Cli:
    def __init__(self):
        self.defaults = config
        self.debug = debug_options

    def _default_args(self, **kwargs):
        args = self.defaults
        if 'debug' in kwargs:
            args.update(self.debug)
        args.update(kwargs)
        args.update(resolve_paths(config))
        args.update(fix_seed(args))
        args.update(get_device(args))
        print(args)

        return Munch(args)

    def check_dataloader(self, **kwargs):
        args = self._default_args(**kwargs)

        iters, vocab = get_iterator(args)
        for batch in iters['train']:
            import ipdb; ipdb.set_trace()  # XXX DEBUG

    def train(self, **kwargs):
        args = self._default_args(**kwargs)

        train(args)

        wait_for_key()

    def evaluate(self, **kwargs):
        args = self._default_args(**kwargs)

        evaluate(args)

        wait_for_key()

    def interactive(self, **kwargs):
        with suppress_stdout():
            args = self._default_args(**kwargs)

            ans = interactive(args)

        print(ans)

    def infer(self, **kwargs):
        args = self._default_args(**kwargs)

        infer(args)


def resolve_paths(config):
    paths = [k for k in config.keys() if k.endswith('_path')]
    res = {}
    root = Path('../').resolve()
    for path in paths:
        res[path] = root / config[path]

    return res


def fix_seed(args):
    if 'random_seed' not in args:
        args['random_seed'] = 0
    random.seed(args['random_seed'])
    np.random.seed(args['random_seed'])
    torch.manual_seed(args['random_seed'])
    torch.cuda.manual_seed_all(args['random_seed'])
    return args


def get_device(args):
    if hasattr(args, 'device'):
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return {'device': device}


if __name__ == "__main__":
    Fire(Cli)

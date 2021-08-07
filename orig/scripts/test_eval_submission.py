import os
import json
import argparse

import numpy as np

from eval_submission import main, open_data


def parse_arg():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-y', '--hypo', default='~/projects/vtt/data/hypo.json', type=str)
    parser.add_argument('-g', '--gt', default='~/projects/vtt/data/QA_train_set_s1s2_1011.json', type=str)
    parser.add_argument('-p', '--prob', default=0.7, type=float)
    args = parser.parse_args()

    return args


def test():
    args = parse_arg()

    gt = open_data(args.gt)
    sample_hypo = make_sample_data(gt, args.prob)
    args.hypo = '/'.join([*args.hypo.split('/')[:-1], 'hypo_sample.json'])

    args.hypo = os.path.expanduser(args.hypo)
    with open(args.hypo, 'w') as f:
        json.dump(sample_hypo, f, indent=4)

    main(args)


def make_sample_data(gt, prob=0.5):
    length = len(list(gt.keys()))
    sample = np.random.binomial(1, prob, length)
    sample = [{'qid': gt['qid'], 'correct_idx': gt['correct_idx']} if p > 0 else
              {'qid': gt['qid'], 'correct_idx': (gt['correct_idx'] + 1) % 5}
                                                   for gt, p in zip(gt.values(), sample)]
    return sample


if __name__ == "__main__":
    test()

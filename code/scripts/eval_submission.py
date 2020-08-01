import json
import os
import argparse
from collections import defaultdict

from tqdm import tqdm


def parse_arg():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-y', '--hypo', default='~/projects/vtt/data/AnotherMissOh/AnotherMissOh_QA/answers.json', type=str)
    parser.add_argument('-g', '--gt', default='~/projects/vtt/data/AnotherMissOh/AnotherMissOh_QA/AnotherMissOhQA_test_set_with_gt.json', type=str)
    args = parser.parse_args()

    return args


def main(args):
    # building qid: row dict
    hypo = open_data(args.hypo)
    gt = open_data(args.gt)

    hypo_keys = set(hypo.keys())
    gt_keys = set(gt.keys())
    assert not (gt_keys - hypo_keys), print("Keys missing: {}".format(gt_keys - hypo_keys))

    gt_dicts = divide_with_key(gt, 'videoType')
    accs = {k: get_acc(hypo, v) for k, v in gt_dicts.items()}
    accs['total'] = [sum(v[0] for v in accs.values()), sum(v[1] for v in accs.values())]
    keys = sorted(list(accs.keys()))

    for k in keys:
        v = accs[k]
        print("{}_accuracy: {}".format(k, v[0] / v[1]))

    return accs


def open_data(path):
    path = os.path.expanduser(path)
    assert os.path.isfile(path), print("file does not exist: {}".format(path))
    with open(path, 'r') as f:
        data = json.load(f)
        if isinstance(data, list):
            data = {row['qid']: row for row in data}
    return data


def divide_with_key(dt, key):
    res = defaultdict(dict)
    for k, v in dt.items():
        res[v[key]][k] = v
    return res


def get_acc(hypo, gt):
    gt_keys = list(gt.keys())
    N = len(gt_keys)
    acc = [float(hypo[k]['correct_idx'] == gt[k]['correct_idx']) for k in gt_keys]
    return sum(acc), N


if __name__ == "__main__":
    args = parse_arg()
    main(args)

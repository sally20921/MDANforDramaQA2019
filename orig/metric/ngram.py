from collections import defaultdict

from .pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from .pycocoevalcap.bleu.bleu import Bleu
from .pycocoevalcap.meteor.meteor import Meteor
from .pycocoevalcap.rouge.rouge import Rouge
from .pycocoevalcap.cider.cider import Cider

from ignite.metrics.metric import Metric
from ignite.exceptions import NotComputableError

from utils import to_string


class Ngram(Metric):
    ngrams = ['bleu', 'meteor', 'rouge', 'meteor']

    def __init__(self, args, vocab, output_transform=lambda x: x):
        super(Ngram, self).__init__(output_transform)

        self.vocab = vocab

        self.tokenizer = PTBTokenizer()
        scorers = {
            'bleu': (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            'meteor': (Meteor(),"METEOR"),
            'rouge': (Rouge(), "ROUGE_L"),
            'cider': (Cider(), "CIDEr")
        }
        self.scorers = [v for k, v in scorers.items() if k in args.metrics]

    @staticmethod
    def default_transform(x):
        return (x[3], x[4])

    def reset(self):
        self._data = defaultdict(lambda: 0)
        self._num_ex = 0

    def format_string(self, x):
        x = to_string(self.vocab, x)
        return {str(i): [v] for i, v in enumerate(x)}

    def update(self, output):
        y_pred, y = output
        num_ex = y_pred.shape[0]
        gts = self.format_string(y_pred)
        res = self.format_string(y)

        for scorer, method in self.scorers:
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, m in zip(score, method):
                    self._data[m] += sc * num_ex
            else:
                self._data[method] += score * num_ex
        self._num_ex += num_ex

    def compute(self):
        if self._num_ex == 0:
            raise NotComputableError(
                'Loss must have at least one example before it can be computed.')
        return {k: v * 100 / self._num_ex for k, v in self._data.items()}
    # *100 for bleu, rouge score percentages

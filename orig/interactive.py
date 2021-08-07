import torch
import numpy as np
from rouge import Rouge
from torchtext.data.batch import Batch

from ckpt import get_model_ckpt
from utils import prepare_batch
from data.dataset import get_tokenizer


def interactive(args):
    args, model, iters, vocab, ckpt_available = get_model_ckpt(args)

    vid = args.get('vid', 's02e09_04_153')
    question = args.get('question', 'What does Monica place on the tablo?')

    batch = get_interactive_batch(args, vid, question, iters)
    with torch.no_grad():
        net_inputs, target = prepare_batch(args, batch, model.vocab)
        y_pred = model(**net_inputs)
        top_1 = y_pred.argmax(dim=-1).item()
        top_0_ans = net_inputs['answers'].squeeze(0)[top_1]
    ans = to_sent(vocab, top_0_ans)

    return ans


def to_sent(vocab, ids):
    ids = [i.item() for i in ids if i.item() not in vocab.special_ids]
    sent = [vocab.itos[ids[i]] for i in range(len(ids))]
    return ' '.join(sent).strip()


def get_interactive_batch(args, vid, question, iters):
    batch = []
    for key, it in iters.items():
        for ex in it.it.dataset.examples:
            if ex.vid == vid:
                batch.append((key, ex))
    tokenizer = get_tokenizer(args)
    ex = batch[check_que_sim(tokenizer, question, batch)]
    key, ex = ex
    ex = Batch([ex], iters[key].it.dataset)
    ex.images = iters[key].get_image(ex.vid)

    return ex


def check_que_sim(tokenizer, question, batch):
    ques = [b[1].que for b in batch]
    ques = [' '.join(b) for b in ques]
    question = ' '.join(question)
    question = [question for i in range(len(batch))]
    r = Rouge()
    scores = r.get_scores(question, ques, avg=False)
    scores = [s['rouge-2']['f'] for s in scores]
    idx = np.array(scores).argmax()
    return idx

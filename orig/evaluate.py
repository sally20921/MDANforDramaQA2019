import torch
from ignite.engine.engine import Engine, State, Events
from metric.stat_metric import StatMetric

from ckpt import get_model_ckpt
from model import get_model
from loss import get_loss
from logger import log_results_cmd

from utils import prepare_batch
from metric import get_metrics


def get_evaluator(args, model, loss_fn, metrics={}):
    def _inference(evaluator, batch):
        model.eval()
        with torch.no_grad():
            net_inputs, target = prepare_batch(args, batch, model.vocab)
            if net_inputs['subtitle'].nelement() == 0:
                import ipdb; ipdb.set_trace()  # XXX DEBUG
            y_pred = model(**net_inputs)
            batch_size = y_pred.shape[0]
            loss, stats = loss_fn(y_pred, target)
            return loss.item(), stats, batch_size, y_pred, target  # TODO: add false_answer metric

    engine = Engine(_inference)

    metrics = {**metrics, **{
        'loss': StatMetric(output_transform=lambda x: (x[0], x[2])),
        'top1_acc': StatMetric(output_transform=lambda x: ((x[3].argmax(dim=-1) == x[4]).float().mean().item(), x[2]))
    }}
    if hasattr(loss_fn, 'get_metric'):
        metrics = {**metrics, **loss_fn.get_metric()}

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def evaluate_once(evaluator, iterator):
    evaluator.run(iterator)
    return evaluator.state


def evaluate(args):
    args, model, iters, vocab, ckpt_available = get_model_ckpt(args)
    if ckpt_available:
        print("loaded checkpoint {}".format(args.ckpt_name))
    loss_fn = get_loss(args, vocab)

    metrics = get_metrics(args, vocab)
    evaluator = get_evaluator(args, model, loss_fn, metrics)

    state = evaluate_once(evaluator, iterator=iters['val'])
    log_results_cmd('valid/epoch', state, 0)

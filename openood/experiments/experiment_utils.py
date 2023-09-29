import torch
import openood.utils.comm as comm
import numpy as np

def save_metrics(loss_avg):
  all_loss = comm.gather(loss_avg)
  total_losses_reduced = np.mean([x for x in all_loss])

  return total_losses_reduced

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * \
                (1 + np.cos(step / total_steps * np.pi))

def save_arr_to_dir(arr, dir):
  with open(dir, 'wb') as f:
    np.save(f, arr)

def msp_postprocess(logits):
  score = torch.softmax(logits, dim=1)
  conf, pred = torch.max(score, dim=1)
  return pred, conf

def print_all_metrics(metrics):
    [fpr, auroc, aupr_in, aupr_out, accuracy] \
      = metrics
    print('FPR@95: {:.2f}, AUROC: {:.2f}'.format(100 * fpr, 100 * auroc),
        end=' ',
        flush=True)
    print('AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}'.format(
      100 * aupr_in, 100 * aupr_out),
        flush=True)
    print('ACC: {:.2f}'.format(accuracy * 100), flush=True)
    print(u'\u2500' * 70, flush=True)
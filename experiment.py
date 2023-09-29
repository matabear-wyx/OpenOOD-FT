from openood.utils import config
from openood.datasets import get_dataloader, get_ood_dataloader
from openood.networks import get_network
from openood.evaluators import get_evaluator

config_files = [
    './configs/finetuning/full-ft.yml',
]
config = config.Config(*config_files)

# get dataloader
id_loader_dict = get_dataloader(config)
ood_loader_dict = get_ood_dataloader(config)
# init network
net = get_network(config.network).cuda()
# init ood evaluator
evaluator = get_evaluator(config)

# Full FT
import torch
import numpy as np

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * \
                (1 + np.cos(step / total_steps * np.pi))

train_loader = id_loader_dict['train']
test_loader = id_loader_dict['test']
val_loader = id_loader_dict['val']

optimizer = torch.optim.SGD(
        net.parameters(),
        config.optimizer.lr,
        momentum=config.optimizer.momentum,
        weight_decay=config.optimizer.weight_decay,
        nesterov=True,)
scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            config.optimizer.num_epochs * len(train_loader),
            1,
            1e-6 / config.optimizer.lr,
        ),
    )

# Linear Probing
import torch
import numpy as np

for param in net.parameters():
    param.requires_grad = False

for param in net.fc.parameters():
    param.requires_grad = True

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * \
                (1 + np.cos(step / total_steps * np.pi))

train_loader = id_loader_dict['train']
test_loader = id_loader_dict['test']
val_loader = id_loader_dict['val']

config.optimizer.num_epochs = 60

optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, net.parameters()),
        config.optimizer.lr,
        momentum=config.optimizer.momentum,
        weight_decay=config.optimizer.weight_decay,
        nesterov=True,)
scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            config.optimizer.num_epochs * len(train_loader),
            1,
            1e-6 / config.optimizer.lr,
        ),
    )

import openood.utils.comm as comm
import numpy as np
def save_metrics(loss_avg):
  all_loss = comm.gather(loss_avg)
  total_losses_reduced = np.mean([x for x in all_loss])

  return total_losses_reduced

from openood.utils import setup_logger
config.save_output = True
setup_logger(config)

from openood.recorders import get_recorder

recorder = get_recorder(config)

from openood.trainers import get_trainer
from openood.evaluators import get_evaluator
trainer = get_trainer(net, train_loader, val_loader, config)
config.evaluator.name = 'base'
evaluator = get_evaluator(config)

print('Start training...', flush=True)
for epoch_idx in range(1, config.optimizer.num_epochs + 1):
  '''
  # For LP-FT
  if epoch_idx == 1:
    for param in net.parameters():
      param.requires_grad = False
    for param in net.fc.parameters():
      param.requires_grad = True
  elif epoch_idx == 31:
    for param in net.parameters():
      param.requires_grad = True
  '''
  # For Surgical-FT
  if epoch_idx == 1:
    for param in net.parameters():
      param.requires_grad = False
    for param in net.fc.parameters():
      param.requires_grad = True
  elif epoch_idx == 41:
    for param in net.parameters():
      param.requires_grad = True
  # train and eval the model
  net, train_metrics = trainer.train_epoch(epoch_idx)
  val_metrics = evaluator.eval_acc(net, val_loader, None, epoch_idx)
  # save model and report the result
  recorder.save_model(net, val_metrics)
  recorder.report(train_metrics, val_metrics)
recorder.summary()
print(u'\u2500' * 70, flush=True)

# evaluate on test set
print('Start testing...', flush=True)
test_metrics = evaluator.eval_acc(net, test_loader)
print('\nComplete Evaluation, accuracy {:.2f}'.format(
   100.0 * test_metrics['acc']),
   flush=True)
print('Completed!', flush=True)
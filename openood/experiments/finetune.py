from openood.utils import config
from openood.datasets import get_dataloader, get_ood_dataloader
from openood.networks import get_network
from openood.trainers import get_trainer
from openood.evaluators import get_evaluator
import torch
import openood.utils.comm as comm
import numpy as np
from openood.utils import setup_logger
from openood.recorders import get_recorder

def save_metrics(loss_avg):
  all_loss = comm.gather(loss_avg)
  total_losses_reduced = np.mean([x for x in all_loss])

  return total_losses_reduced

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * \
                (1 + np.cos(step / total_steps * np.pi))

class BaseExperiment:
    def __init__(self, config):
        self.config = config
        self.net = self.init_net()
        self.optimizer = self.init_optimizer()
        self.id_loader_dict = get_dataloader(config)
        self.ood_loader_dict = get_ood_dataloader(config)
        self.train_loader = self.id_loader_dict['train']
        self.test_loader = self.id_loader_dict['test']
        self.val_loader = self.id_loader_dict['val']
        self.trainer = self.init_trainer()
        self.evaluator = self.init_evaluator()
        self.scheduler = self.init_scheduler()
        self.recorder = get_recorder(config)

    def init_net(self):
        net = get_network(self.config.network).cuda()
        return net

    def init_trainer(self):
        trainer = get_trainer(self.net, self.train_loader, self.val_loader, self.config)
        return trainer

    def init_evaluator(self):
        config.evaluator.name = 'base'
        evaluator = get_evaluator(self.config)
        return evaluator

    def init_optimizer(self):
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.net.parameters()),
            self.config.optimizer.lr,
            momentum=self.config.optimizer.momentum,
            weight_decay=self.config.optimizer.weight_decay,
            nesterov=True,
        )
        return optimizer

    def init_scheduler(self):
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                self.config.optimizer.num_epochs * len(self.train_loader),
                1,
                1e-6 / self.config.optimizer.lr,
            ),
        )
        return scheduler

    def train_epoch(self, epoch_idx):
        # train and eval the model
        net, train_metrics = self.trainer.train_epoch(epoch_idx)
        val_metrics = self.evaluator.eval_acc(self.net, self.val_loader, None, epoch_idx)
        # save model and report the result
        self.recorder.save_model(net, val_metrics)
        self.recorder.report(train_metrics, val_metrics)

    def run(self):
        config.save_output = True
        setup_logger(config)
        self.config.evaluator.name = 'base'
        for epoch_idx in range(1, self.config.optimizer.num_epochs + 1):
            self.train_epoch(epoch_idx)
        self.recorder.summary()
        print(u'\u2500' * 70, flush=True)

        # evaluate on test set
        print('Start testing...', flush=True)
        test_metrics = self.evaluator.eval_acc(self.net, self.test_loader)
        print('\nComplete Evaluation, accuracy {:.2f}'.format(
            100.0 * test_metrics['acc']),
            flush=True)
        print('Completed!', flush=True)


class FullFTExperiment(BaseExperiment):
    pass

class LinearProbingExperiment(BaseExperiment):
    def train_epoch(self, epoch_idx):
        if epoch_idx == 1:
            for param in self.net.parameters():
                param.requires_grad = False
            for param in self.net.fc.parameters():
                param.requires_grad = True
            self.optimizer = self.init_optimizer()
        super().train_epoch(epoch_idx)

class LPFTExperiment(BaseExperiment):
    def train_epoch(self, epoch_idx):
        if epoch_idx == 1:
            for param in self.net.parameters():
                param.requires_grad = False
            for param in self.net.fc.parameters():
                param.requires_grad = True
            self.optimizer = self.init_optimizer()
        elif epoch_idx == self.config.optimizer.num_epochs // 2 + 1:
            for param in self.net.parameters():
                param.requires_grad = True
            self.optimizer = self.init_optimizer()

        super().train_epoch(epoch_idx)

class SurgicalFTExperiment(BaseExperiment):
    def train_epoch(self, epoch_idx):
        if epoch_idx == 1:
            for param in self.net.parameters():
                param.requires_grad = False
            for param in self.net.fc.parameters():
                param.requires_grad = True
            self.optimizer = self.init_optimizer()
        elif epoch_idx == (self.config.optimizer.num_epochs * 2) // 3 + 1:
            for param in self.net.parameters():
                param.requires_grad = True
            self.optimizer = self.init_optimizer()

        super().train_epoch(epoch_idx)
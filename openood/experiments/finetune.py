from openood.datasets import get_dataloader, get_ood_dataloader
from openood.networks import get_network
from openood.trainers import get_trainer
from openood.evaluators import get_evaluator
from openood.evaluators.metrics import compute_all_metrics
from .experiment_utils import cosine_annealing, msp_postprocess, print_all_metrics, save_metrics, save_arr_to_dir
from tqdm import tqdm
import numpy as np
import torch
import os.path as osp
import os
from openood.utils import setup_logger
from openood.recorders import get_recorder

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
        self.results = None
        self.postprocess_results = None
        self.save_root = f'./results/{self.config.exp_name}'

    def init_net(self):
        net = get_network(self.config.network).cuda()
        return net

    def init_trainer(self):
        trainer = get_trainer(self.net, self.train_loader, self.val_loader, self.config)
        return trainer

    def init_evaluator(self):
        self.config.evaluator.name = 'base'
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
        self.config.save_output = True
        setup_logger(self.config)
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

    def id_result(self):
        self.config.evaluator.name = 'ood'
        self.net.eval()
        modes = ['test', 'val']
        for mode in modes:
            dl = self.id_loader_dict[mode]
            dataiter = iter(dl)

            logits_list = []
            feature_list = []
            label_list = []

            for i in tqdm(range(1,
                                len(dataiter) + 1),
                          desc='Extracting results...',
                          position=0,
                          leave=True):
                batch = next(dataiter)
                data = batch['data'].cuda()
                label = batch['label']
                with torch.no_grad():
                    logits_cls, feature = self.net(data, return_feature=True)
                logits_list.append(logits_cls.data.to('cpu').numpy())
                feature_list.append(feature.data.to('cpu').numpy())
                label_list.append(label.numpy())

            logits_arr = np.concatenate(logits_list)
            feature_arr = np.concatenate(feature_list)
            label_arr = np.concatenate(label_list)

            logits_dir = os.path.join(self.save_root, 'id')
            os.makedirs(logits_dir, exist_ok=True)

            save_arr_to_dir(logits_arr, osp.join(self.save_root, 'id', f'{mode}_logits.npy'))
            save_arr_to_dir(feature_arr, osp.join(self.save_root, 'id', f'{mode}_feature.npy'))
            save_arr_to_dir(label_arr, osp.join(self.save_root, 'id', f'{mode}_labels.npy'))

    def ood_result(self):
        self.net.eval()
        ood_splits = ['nearood', 'farood']
        for ood_split in ood_splits:
            for dataset_name, ood_dl in self.ood_loader_dict[ood_split].items():
                dataiter = iter(ood_dl)

                logits_list = []
                feature_list = []
                label_list = []

                for i in tqdm(range(1,
                                    len(dataiter) + 1),
                              desc='Extracting results...',
                              position=0,
                              leave=True):
                    batch = next(dataiter)
                    data = batch['data'].cuda()
                    label = batch['label']

                    with torch.no_grad():
                        logits_cls, feature = self.net(data, return_feature=True)
                    logits_list.append(logits_cls.data.to('cpu').numpy())
                    feature_list.append(feature.data.to('cpu').numpy())
                    label_list.append(label.numpy())

                logits_arr = np.concatenate(logits_list)
                feature_arr = np.concatenate(feature_list)
                label_arr = np.concatenate(label_list)

                logits_dir = os.path.join(self.save_root, ood_split)
                os.makedirs(logits_dir, exist_ok=True)

                save_arr_to_dir(logits_arr, osp.join(self.save_root, ood_split, f'{dataset_name}_logits.npy'))
                save_arr_to_dir(feature_arr, osp.join(self.save_root, ood_split, f'{dataset_name}_feature.npy'))
                save_arr_to_dir(label_arr, osp.join(self.save_root, ood_split, f'{dataset_name}_labels.npy'))

    def load_result(self):
        results = dict()
        # for id
        modes = ['val', 'test']
        results['id'] = dict()
        for mode in modes:
            results['id'][mode] = dict()
            results['id'][mode]['feature'] = np.load(osp.join(self.save_root, 'id', f'{mode}_feature.npy'))
            results['id'][mode]['logits'] = np.load(osp.join(self.save_root, 'id', f'{mode}_logits.npy'))
            results['id'][mode]['labels'] = np.load(osp.join(self.save_root, 'id', f'{mode}_labels.npy'))

        # for ood
        split_types = ['nearood', 'farood']
        for split_type in split_types:
            results[split_type] = dict()
            dataset_names = self.config['ood_dataset'][split_type].datasets
            for dataset_name in dataset_names:
                results[split_type][dataset_name] = dict()
                results[split_type][dataset_name]['feature'] = np.load(
                    osp.join(self.save_root, split_type, f'{dataset_name}_feature.npy'))
                results[split_type][dataset_name]['logits'] = np.load(
                    osp.join(self.save_root, split_type, f'{dataset_name}_logits.npy'))
                results[split_type][dataset_name]['labels'] = np.load(
                    osp.join(self.save_root, split_type, f'{dataset_name}_labels.npy'))
        self.results = results

    def postprocess(self):
        postprocess_results = dict()
        # id
        modes = ['val', 'test']
        postprocess_results['id'] = dict()
        for mode in modes:
            pred, conf = msp_postprocess(torch.from_numpy(self.results['id'][mode]['logits']))
            pred, conf = pred.numpy(), conf.numpy()
            gt = self.results['id'][mode]['labels']
            postprocess_results['id'][mode] = [pred, conf, gt]

        # ood
        split_types = ['nearood', 'farood']
        for split_type in split_types:
            postprocess_results[split_type] = dict()
            dataset_names = self.config['ood_dataset'][split_type].datasets
            for dataset_name in dataset_names:
                pred, conf = msp_postprocess(torch.from_numpy(self.results[split_type][dataset_name]['logits']))
                pred, conf = pred.numpy(), conf.numpy()
                gt = self.results[split_type][dataset_name]['labels']
                gt = -1 * np.ones_like(gt)  # hard set to -1 here
                postprocess_results[split_type][dataset_name] = [pred, conf, gt]

        self.postprocess_results = postprocess_results

    def eval_ood(self):
        [id_pred, id_conf, id_gt] = self.postprocess_results['id']['test']
        split_types = ['nearood', 'farood']

        for split_type in split_types:
            metrics_list = []
            print(f"Performing evaluation on {split_type} datasets...")
            dataset_names = self.config['ood_dataset'][split_type].datasets

            for dataset_name in dataset_names:
                [ood_pred, ood_conf, ood_gt] = self.postprocess_results[split_type][dataset_name]
                pred = np.concatenate([id_pred, ood_pred])
                conf = np.concatenate([id_conf, ood_conf])
                label = np.concatenate([id_gt, ood_gt])
                print(f'Computing metrics on {dataset_name} dataset...')

                ood_metrics = compute_all_metrics(conf, label, pred)
                print_all_metrics(ood_metrics)
                metrics_list.append(ood_metrics)
            print('Computing mean metrics...', flush=True)
            metrics_list = np.array(metrics_list)
            metrics_mean = np.mean(metrics_list, axis=0)

            print_all_metrics(metrics_mean)

    def evaluate(self):
        self.id_result()
        self.ood_result()
        self.load_result()
        self.postprocess()
        self.eval_ood()


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
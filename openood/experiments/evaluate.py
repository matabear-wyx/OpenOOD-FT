from openood.datasets import get_dataloader, get_ood_dataloader
from tqdm import tqdm
import numpy as np
import torch
import os.path as osp
import os
from openood.evaluators.metrics import compute_all_metrics

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

class Evaluator:
    def __init__(self, config, net):
        self.config = config
        self.save_root = f'./results/{config.exp_name}'
        self.net = net
        self.id_loader_dict = get_dataloader(config)
        self.ood_loader_dict = get_ood_dataloader(config)
        self.results = None
        self.postprocess_results = None

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
        if self.results == None:
            raise Exception()
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


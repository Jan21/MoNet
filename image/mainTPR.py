import time
import os.path as osp
import argparse

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (graclus, max_pool, global_mean_pool)
from conv import TPRConv
from collections import Counter

parser = argparse.ArgumentParser(description='superpixel MNIST')
parser.add_argument('--dataset', default='MNIST', type=str)
parser.add_argument('--device_idx', default=0, type=int)
parser.add_argument('--kernel_size', default=25, type=int)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=0.99)
parser.add_argument('--decay_step', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

args.data_fp = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                        args.dataset)
device = torch.device('cuda', args.device_idx)

# deterministic
torch.manual_seed(args.seed)
cudnn.benchmark = False
cudnn.deterministic = True

def remove_self_loops(data):

    #filtered_test_ix_arr = []
    #test_indices = data.edge_index.T
    #for i,x in enumerate(test_indices):
    #    if x[0] != x[1]:
    #        filtered_test_ix_arr.append(x)
    #data.edge_index = torch.stack(filtered_test_ix_arr,dim=0).T
    return data


#xs,ys = Counter(),Counter()

#threshs = [0.2,0.4,0.6,0.8]

# for tup in test_dataset.data.edge_attr:
#     x = tup[0]
#     y = tup[1]
#     x_bin = 4
#     y_bin = 4
#     if x< threshs[0]:
#        x_bin = 0
#     elif x < threshs[1]:
#         x_bin = 1
#     elif x < threshs[2]:
#         x_bin = 2
#     elif x < threshs[3]:
#         x_bin = 3
#
#     if y < threshs[0]:
#        y_bin = 0
#     elif y < threshs[1]:
#         y_bin = 1
#     elif y < threshs[2]:
#         y_bin = 2
#     elif y < threshs[3]:
#         y_bin = 3
#     xs[x_bin] += 1
#     ys[y_bin] += 1

def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))


import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import random_split
from torchvision import datasets, transforms
from torch.nn import functional as F
from pytorch_lightning.callbacks import EarlyStopping
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
import os
import os.path as osp

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.loggers import LightningLoggerBase


class TPRGNNet(pl.LightningModule):
    def __init__(self,trial):
        super(TPRGNNet, self).__init__()
        self.TPR_filler_dim = 10
        self.reduce_for_TPR = False
        self.add_nonlin_before_TPR = False
        self.do_TPR_in_update = False
        self.out1 = 64
        self.out2 = 64
        self.out3 = 64
        self.linout = trial.suggest_categorical('linout', [128])
        self.nonlin_ix = trial.suggest_categorical('activation_type', [1,0])
        print("*"*50)
        print('reduce_for_TPR: ',self.reduce_for_TPR)
        print('add_nonlin_before_TPR: ',self.add_nonlin_before_TPR)
        print('do_TPR_in_update: ', self.do_TPR_in_update)
        print('out1, out2, out3, linout: ',self.out1, self.out3, self.out3, self.linout)
        print('nonlin_ix:', self.nonlin_ix)
        self.nonlin = [F.elu,F.relu,F.gelu][self.nonlin_ix]
        self.conv1 = TPRConv(1, self.out1, dim=2, TPR_filler_dim=self.TPR_filler_dim, reduce_for_TPR=self.reduce_for_TPR, do_TPR_in_update=self.do_TPR_in_update)
        self.conv2 = TPRConv(self.out1*2, self.out2, dim=2, TPR_filler_dim=self.TPR_filler_dim, reduce_for_TPR=self.reduce_for_TPR, do_TPR_in_update=self.do_TPR_in_update)
        self.conv3 = TPRConv(self.out2*2, self.out3, dim=2, TPR_filler_dim=self.TPR_filler_dim, reduce_for_TPR=self.reduce_for_TPR, do_TPR_in_update=self.do_TPR_in_update)
        self.fc1 = torch.nn.Linear(self.out3*2, self.linout)
        self.fc2 = torch.nn.Linear(self.linout, 10)

    def forward(self, data):
        data.x = self.nonlin(self.conv1(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.edge_attr = None
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data.x = self.nonlin(self.conv2(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data.x = self.nonlin(self.conv3(data.x, data.edge_index, data.edge_attr))

        x = global_mean_pool(data.x, data.batch)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        logits = self.forward(train_batch)
        loss = self.cross_entropy_loss(logits, train_batch.y)
        pred = logits.max(1)[1]
        accuracy = pred.eq(train_batch.y).sum().item()
        logs = {"train_loss": loss, "accuracy":accuracy}
        return {"loss": loss, "log": logs}

    def validation_step(self, val_batch, batch_idx):
        logits = self.forward(val_batch)
        loss = self.cross_entropy_loss(logits, val_batch.y)
        pred = logits.max(1)[1]
        accuracy = pred.eq(val_batch.y)
        return {"val_loss": loss, "val_accuracy": accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_accuracy = torch.cat([x["val_accuracy"] for x in outputs]).sum().item() / len(self.test_dataset)
        tensorboard_logs = {"val_loss": avg_loss, 'accuracy':avg_accuracy}
        print('----------ACCURACY: ',avg_accuracy)
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    args.decay_step,
                                                    gamma=args.lr_decay)
        return [optimizer],[scheduler]

    def prepare_data(self):
        self.train_dataset = MNISTSuperpixels(args.data_fp, True, pre_transform=T.Compose([remove_self_loops, T.Polar()]))
        self.test_dataset = MNISTSuperpixels(args.data_fp, False, pre_transform=T.Compose([remove_self_loops, T.Polar()]))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=64, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=64)


#model = TPRGNNet()
#trainer = pl.Trainer(gpus=1, progress_bar_refresh_rate=50)
#trainer.fit(model)


class CustomPyTorchLightningPruningCallback(PyTorchLightningPruningCallback):
    def on_epoch_end(self, epoch, logs=None):
        # type: (int, Optional[Dict[str, float]]) -> None

        logs = logs or {}
        current_score = epoch.logger.metrics[-1][self._monitor]
        if current_score is None:
            return
        self._trial.report(current_score, step=epoch.current_epoch)
        if current_score < 0.5:
            message = "Trial was pruned at epoch {}. Lower than 0.5.".format(epoch.current_epoch)
            raise optuna.exceptions.TrialPruned(message)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch.current_epoch)
            raise optuna.exceptions.TrialPruned(message)



PERCENT_TEST_EXAMPLES = 0.1
BATCHSIZE = 64
CLASSES = 10
EPOCHS = 300
DIR = os.getcwd()
MODEL_DIR = os.path.join(DIR, "result")


class DictLogger(LightningLoggerBase):
    """PyTorch Lightning `dict` logger."""

    def __init__(self, version):
        super(DictLogger, self).__init__()
        self.name = 'test'
        self.metrics = []
        self._version = version

    def log_hyperparams(self, params):
        print("--" * 30)
        print("params:",params)

    def name(self, name):
        print("--" * 30)
        print("name of the experiment:", name)

    def experiment(self, exp):
        print("--" * 30)
        print("New experiment is starting")

    def log_metrics(self, metrics, step=None):
        self.metrics.append(metrics)

    @property
    def version(self):
        return self._version


def objective(trial):
    # PyTorch Lightning will try to restore model parameters from previous trials if checkpoint
    # filenames match. Therefore, the filenames for each trial must be made unique.

    # Check duplication and skip if it's detected.
    for t in trial.study.trials:
        if t.state != optuna.structs.TrialState.COMPLETE:
            continue

        if t.params == trial.params:
            return t.value

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "trial_{}".format(trial.number)), monitor="accuracy"
    )

    # The default logger in PyTorch Lightning writes to event files to be consumed by
    # TensorBoard. We create a simple logger instead that holds the log in memory so that the
    # final accuracy can be obtained after optimization. When using the default logger, the
    # final accuracy could be stored in an attribute of the `Trainer` instead.
    logger = DictLogger(trial.number)

    PruningCallback = CustomPyTorchLightningPruningCallback(trial, monitor="accuracy")
    trainer = pl.Trainer(
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        max_epochs=EPOCHS,
        gpus=1 if torch.cuda.is_available() else None,
        early_stop_callback=PruningCallback,
        progress_bar_refresh_rate=100
    )
    model = TPRGNNet(trial)
    trainer.fit(model)

    return logger.metrics[-1]["accuracy"]


pruner = optuna.pruners.MedianPruner()

study = optuna.create_study(direction="maximize", pruner=pruner)
study.optimize(objective, n_trials=1, timeout=600)

print("Number of finished trials: {}".format(len(study.trials)))


print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

#shutil.rmtree(MODEL_DIR)
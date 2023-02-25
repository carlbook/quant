#!/home/carl/miniconda3/envs/trading_pytorch/bin/python
import os
import sys
import csv
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime as dt
sys.path.extend(['/home/carl/trading/quant'])  # Must extend path for dataloaders when this file is not in parent dir
from dataloaders import SequenceCRSPdsf62InMemory
from custom_pt_obj import configure_opt_for_model_wt_decay

device = 'cuda:0'  # BusID 11 (0b) = 3080 Ti, BusID 4 = 1050 Ti . See lspci and nvidia-settings.
os.chdir(os.path.dirname(os.path.realpath(__file__)))


class StocksNet(nn.Module):

    def __init__(self, sv_channels, sv_seq_len, num_trgt_classes, with_sp500=False):
        super(StocksNet, self).__init__()
        self.with_sp500 = with_sp500
        sp500_input_width = 256 if with_sp500 else 0
        cl_vs_sma_input = 512
        final_lyr = 8192
        self.conv_stack = nn.Sequential(
            nn.Conv1d(sv_channels, 128, 3, bias=True, padding='same'),
            nn.Mish(inplace=True),  # inplace=True is probably not valid if skip-connections are used
            nn.Conv1d(128, 224, 3, bias=True, padding='same'),
            nn.Mish(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(224, 320, 3, bias=True, padding='same'),
            nn.Mish(inplace=True),
            nn.Conv1d(320, 512, 3, bias=True, padding='same'),
            nn.Mish(inplace=True),
            nn.MaxPool1d(2),
            nn.Flatten(),  # start_dim=1 default arg preserves batch dimension
            nn.Dropout(p=0.5))
        self.cl_hist_net = nn.Sequential(
            nn.Linear(3, cl_vs_sma_input, bias=True),
            nn.Mish(inplace=True),
            nn.Dropout(p=0.5))
        self.final_fc_net = nn.Sequential(
            nn.Linear(sv_seq_len // 2 // 2 * 512 + cl_vs_sma_input + sp500_input_width, final_lyr),
            nn.Mish(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(final_lyr, num_trgt_classes))
        if with_sp500:
            self.sp500_net = nn.Sequential(
                nn.Linear(1, sp500_input_width, bias=True),
                nn.Mish(inplace=True),
                nn.Dropout(p=0.2))
            self.fpass_func = self.__forward_with_sp500
        else:
            self.fpass_func = self.__forward

    def forward(self, inputs_list):
        # Expected inputs_list ordering is svect_seq, cl_hist_dct_coeffs, sp500_onehot
        return self.fpass_func(inputs_list)

    def __forward(self, inputs_list):
        conv_features = self.conv_stack(inputs_list[0])
        cl_hist_features = self.cl_hist_net(inputs_list[1])
        fc_inputs = torch.cat((conv_features, cl_hist_features), dim=1)
        logits = self.final_fc_net(fc_inputs)
        return logits

    def __forward_with_sp500(self, inputs_list):
        conv_features = self.conv_stack(inputs_list[0])
        cl_hist_features = self.cl_hist_net(inputs_list[1])
        sp500_features = self.sp500_net(inputs_list[2])
        fc_inputs = torch.cat((conv_features, cl_hist_features, sp500_features), dim=1)
        logits = self.final_fc_net(fc_inputs)
        return logits


class Model:

    def __init__(self, target, sufx, trn_seq=None, test_seq=None, include_sp5=False, trn_batch=1024, test_batch=1024,
                 num_epochs=5, patience=None, use_amp=False, trn_loss_sample_rate=0.1):
        self.target = target
        self.config_suffix = sufx
        self.trn_seq = trn_seq
        self.test_seq = test_seq
        self.include_sp5 = include_sp5
        self.trn_batch = trn_batch
        self.test_batch = test_batch
        self.num_epochs = num_epochs
        self.patience = patience  # number of epochs befor early stopping is triggered
        self.use_amp = use_amp
        now_str = dt.now().strftime("%Y%m%d_%H%M")
        self.trn_log_file = f'trn_log_{sufx}_{target}{"_sp500" if include_sp5 else ""}_{now_str}.csv'
        self.best_mdl_file = f'best_model_{sufx}_{target}{"_sp500" if include_sp5 else ""}.pt'
        self.trn_loss_smpl_rate = trn_loss_sample_rate  # fraction of tot num batches at which to sample train loss
        self.num_trn_samples = 0
        self.num_test_samples = 0
        self.num_full_trn_batches = 0
        self.base_lr = None
        self.lbl_smth = {'cc': 0.10, 'oc': 0.10, 'md': 0.10}[target]
        self.linear_lyr_w_decay = 0.01
        self.__build_model()

    def __build_model(self):
        # Retrieve the sv and closing price dct coeff dimensions from the underlying as-loaded dataset
        available_dset = self.test_seq if self.test_seq else self.trn_seq
        sv_channels, sv_seq_len = available_dset.attrs['vector_dims']
        self.num_output_classes = available_dset.num_classes
        self.model = StocksNet(
            sv_channels,
            sv_seq_len,
            self.num_output_classes,
            with_sp500=self.include_sp5
        ).to(device)

        # LOSS FUNCTION
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=self.lbl_smth)  # expects integer targets

        # OPTIMIZER - note that only the Linear layer weights should be whitelisted for decay
        optimizer_groups = configure_opt_for_model_wt_decay(self.model, weight_decay=self.linear_lyr_w_decay)
        self.optimizer = torch.optim.AdamW(optimizer_groups, lr=self.get_init_lrn_rate(self.num_output_classes))

        # GRAD SCALER FOR MIXED PRECISION
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # LEARNING RATE SCHEDULER
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda ep: 1.13 if ep < 7 else 0.96)

        if self.trn_seq:
            self.num_trn_samples = len(self.trn_seq)
            self.num_full_trn_batches = self.num_trn_samples // self.trn_batch
            self.trn_data = DataLoader(
                self.trn_seq,
                batch_size=self.trn_batch,
                shuffle=True,
                drop_last=True,
                prefetch_factor=8,
                num_workers=4,
                pin_memory=True,
                pin_memory_device=device)
            self.trn_loss_batches_smpl_interval = int(self.trn_loss_smpl_rate * self.num_full_trn_batches)
            with open(self.trn_log_file, 'a', newline='') as csvfile:
                wr = csv.DictWriter(csvfile, fieldnames=('Epoch', 'Seconds', 'Est Avg Trn Loss', 'Val Loss', 'Val Acc'))
                wr.writeheader()
        else: print('No training data was passed. Model is available for inference only')
        if self.test_seq:
            self.num_test_samples = len(self.test_seq)
            self.test_data = DataLoader(
                self.test_seq,
                batch_size=self.test_batch,
                shuffle=False,
                drop_last=False,
                num_workers=4)
        else: print('No test data was passed. Test / Validation metrics will not be computed')

    def get_init_lrn_rate(self, num_classes):
        if 'vn' in self.config_suffix: self.base_lr = {4: 8e-4, 10: 1e-3, 14: 1.3e-3}[num_classes]
        else: self.base_lr = {4: 8e-4, 10: 1e-3, 14: 1.3e-3}[num_classes]
        return self.base_lr

    def print_model_summary(self):
        print(self.model)
        num_trainable = 0
        print('\nSummary of trainable parameters')
        for name, param in self.model.named_parameters():
            if not param.requires_grad: continue
            num = param.numel()
            print(f'{name:<30}{num}')
            num_trainable += num
        print(f'\nTotal Trainable Parameters: {num_trainable}\n')

    # noinspection PyTypeChecker
    def __train(self):
        running_losses = []
        self.model.train()
        for batch_num, (inputs, targets) in enumerate(self.trn_data):
            self.optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
                inputs[0] = inputs[0].to(device)
                inputs[1] = inputs[1].to(device)
                if self.include_sp5:
                    inputs[2] = inputs[2].to(device)
                targets = targets.to(device)
                pred = self.model(inputs)
                loss = self.loss_fn(pred, targets)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if (batch_num + 1) % self.trn_loss_batches_smpl_interval == 0:
                loss, samples_processed = loss.item(), batch_num * len(inputs[0])
                running_losses.append(loss)
                print(f'trn loss: {loss:>7f}  [{samples_processed:>5d}/{self.num_trn_samples:>5d}]')
                # for name, weight in self.model.named_parameters():
                #     print(f'{name:<30}{weight.min().item():>10f}\t'
                #           f'{weight.max().item():>10f}\t{weight.isnan().any().item()}')
                # print()
        return sum(running_losses) / len(running_losses)

    # noinspection PyTypeChecker
    def test(self):
        num_batches = len(self.test_data)
        self.model.eval()
        test_loss, correct = 0., 0.
        with torch.no_grad():
            for inputs, targets in self.test_data:
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
                    inputs[0] = inputs[0].to(device)
                    inputs[1] = inputs[1].to(device)
                    if self.include_sp5:
                        inputs[2] = inputs[2].to(device)
                    targets = targets.to(device)
                    pred = self.model(inputs)
                    test_loss += self.loss_fn(pred, targets).item()
                correct += (pred.argmax(1) == targets).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= self.num_test_samples
        print(f'Test Metrics:\n  Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')
        return test_loss, correct

    def inference(self, have_labels=True):
        """Sample-level outputs. Set have_labels=False when you don't know the future"""
        loss_func = nn.CrossEntropyLoss(reduction='none')  # NO LABEL SMOOTHING HERE; LOSSES WILL BE SMALLER
        predictions, truth, losses = [], [], []
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in self.test_data:
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
                    inputs[0] = inputs[0].to(device)
                    inputs[1] = inputs[1].to(device)
                    if self.include_sp5:
                        inputs[2] = inputs[2].to(device)
                    pred = self.model(inputs)
                    if have_labels:
                        targets = targets.to(device)
                        losses.append(loss_func(pred, targets).detach())
                        truth.append(targets.detach())
                    predictions.append(pred.detach())
        predictions = torch.cat(predictions).cpu()
        if have_labels:
            truth = torch.cat(truth).cpu()
            losses = torch.cat(losses).cpu()
        return predictions, truth, losses

    def train_model(self):
        if not self.trn_seq:
            print('Cannot train, no dataset was provided')
            return
        self.print_model_summary()
        best_val_loss, best_mdl_val_acc, best_epoch = np.inf, 0.0, 1
        for t in range(1, self.num_epochs + 1):
            print(f'Epoch {t}\n----------------------------------')
            start_dt = dt.now()
            est_avg_trn_loss = self.__train()
            end_dt = dt.now()
            elapsed_sec = f'{(end_dt - start_dt).total_seconds():.0f}'
            print(f'Epoch complete. Elapsed time: {elapsed_sec}s')
            if self.test_seq:
                val_loss, val_acc = self.test()
                self.__log_training_progress(t, elapsed_sec, est_avg_trn_loss, val_loss, val_acc)
                # Best Model is defined in terms of validation loss
                if val_loss < best_val_loss:
                    self.save_model(t, val_loss, val_acc)
                    best_val_loss = val_loss
                    best_mdl_val_acc = val_acc  # Accuracy of model with the lowest loss
                    best_epoch = t
                if t - best_epoch > self.patience:
                    print('Early Stopping triggered')
                    break
            else:
                self.__log_training_progress(t, elapsed_sec, est_avg_trn_loss, '', '')
            self.lr_scheduler.step()
        print(f'\n-----Best Model Summary-----\n'
              f'Epoch: {best_epoch}\n'
              f'Validation Loss: {best_val_loss:.6f}\n'
              f'Validation Accuracy: {(100. * best_mdl_val_acc):>0.1f}%')

    def __log_training_progress(self, epoch, elapsed_sec, est_avg_trn_loss, val_loss, val_acc):
        with open(self.trn_log_file, 'a', newline='') as csvfile:
            wr = csv.DictWriter(csvfile, fieldnames=('Epoch', 'Seconds', 'Est Avg Trn Loss', 'Val Loss', 'Val Acc'))
            wr.writerow({
                'Epoch':            epoch,
                'Seconds':          elapsed_sec,
                'Est Avg Trn Loss': est_avg_trn_loss,
                'Val Loss':         val_loss,
                'Val Acc':          val_acc})

    def save_model(self, current_epoch, val_loss, val_acc):
        torch.save({
            'initial_lrn_rate':         self.base_lr,
            'label_smoothing':          self.lbl_smth,
            'linear_lyr_wght_decay':    self.linear_lyr_w_decay,
            'trn_batch_size':           self.trn_batch,
            'num_output_classes':       self.num_output_classes,
            'include_sp500_input':      self.include_sp5,
            'epoch':                    current_epoch,
            'model_state_dict':         self.model.state_dict(),
            'optimizer_state_dict':     self.optimizer.state_dict(),
            'scaler_state_dict':        self.scaler.state_dict(),
            'lr_sched_state_dict':      self.lr_scheduler.state_dict(),
            'val_loss':                 val_loss,
            'val_acc':                  val_acc
        }, self.best_mdl_file)


def main(args):
    """The dataset used here only has one fold. To run with sp500 input, do './model.py sp500'"""
    include_sp5 = True if 'sp500' in args else False
    sufx = args[1]
    trgt = args[2]
    trn_args = (f'/mnt/data/trading/datasets/CRSPdsf62_trn_{sufx}.hdf5',)
    val_args = (f'/mnt/data/trading/datasets/CRSPdsf62_val_{sufx}.hdf5',)
    dataset_kwargs = {'with_sp500': include_sp5, 'target': trgt, 'num_samples': None}
    trn_seq = SequenceCRSPdsf62InMemory(*trn_args, **dataset_kwargs)
    val_seq = SequenceCRSPdsf62InMemory(*val_args, **dataset_kwargs)
    model = Model(
        trgt,
        sufx,
        trn_seq=trn_seq,
        test_seq=val_seq,
        include_sp5=include_sp5,
        trn_batch=1024,
        test_batch=8192,
        num_epochs=30,
        patience=5,
        use_amp=True)
    model.train_model()


if __name__ == '__main__':
    main(sys.argv)

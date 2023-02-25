#!/home/carl/miniconda3/envs/trading_pytorch/bin/python
import seaborn as sn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from dataloaders import SequenceCRSPdsf62InMemory, InferenceInMemory
from models.m057.model import Model


def m057(sufx, trgt):
    saved_mdl_path = f'/home/carl/trading/quant/models/m057/best_model_{sufx}_{trgt}.pt'
    checkpoint = torch.load(saved_mdl_path)
    include_sp5 = False
    val_args = (f'/mnt/data/trading/datasets/CRSPdsf62_val_{sufx}.hdf5',)
    dataset_kwargs = {'with_sp500': include_sp5, 'target': trgt, 'num_samples': None}
    val_seq = SequenceCRSPdsf62InMemory(*val_args, **dataset_kwargs)
    # this label represents the first greater-than-zero bin
    zero_bin_idx = val_seq.attrs[f'{trgt}_bins'].searchsorted(0, side='right')
    best_m057 = Model(
        trgt,
        sufx,
        test_seq=val_seq,
        include_sp5=include_sp5,
        test_batch=8192)
    best_m057.model.load_state_dict(checkpoint['model_state_dict'])
    # loss, acc = best_m057.test()
    # print(loss)
    # print(checkpoint['val_loss'])
    # print(acc)
    # print(checkpoint['val_acc'])
    preds, acts_ints, losses = best_m057.inference()
    acts_ints = acts_ints.numpy()
    pred_ints = preds.argmax(dim=1).numpy()
    pred_probs = nn.functional.softmax(preds, dim=1).numpy()
    df = pd.DataFrame(data=pred_probs)
    df['Actuals'] = acts_ints
    gr = df.groupby(by='Actuals').sum()
    gr.columns.name = 'Predictions'
    pred_probabilities = gr.divide(gr.sum(axis=1), axis='index')
    loser_density = pred_probs[:, :zero_bin_idx].sum(axis=1)
    gainer_density = pred_probs[:, zero_bin_idx:].sum(axis=1)
    df = pd.DataFrame(
        data=np.vstack((loser_density, gainer_density)).transpose(),
        columns=['Loser Density', 'Gainer Density'])
    df['Actuals'] = acts_ints
    gr = df.groupby(by='Actuals').sum()
    gr.columns.name = 'Predictions'
    gl_probabilities = gr.divide(gr.sum(axis=1), axis='index')
    return preds, pred_ints, acts_ints, pred_probabilities, gl_probabilities


def m057_yahoo(sufx, trgt, data):
    saved_mdl_path = f'/home/carl/trading/quant/models/m057/best_model_{sufx}_{trgt}.pt'
    checkpoint = torch.load(saved_mdl_path)
    c_factor = 3
    zero_bin_idx = data.attrs[f'{trgt}_bins'].searchsorted(0, side='right')
    best_m057 = Model(
        trgt,
        sufx,
        test_seq=data,
        include_sp5=False,
        test_batch=8192)
    best_m057.model.load_state_dict(checkpoint['model_state_dict'])
    preds, acts_ints, losses = best_m057.inference()
    acts_ints = acts_ints.numpy()
    pred_ints = preds.argmax(dim=1).numpy()
    pred_probs = nn.functional.softmax(preds, dim=1).numpy()
    df = pd.DataFrame(data=pred_probs)
    df['Actuals'] = acts_ints
    gr = df.groupby(by='Actuals').sum()
    gr.columns.name = 'Predictions'
    pred_probabilities = gr.divide(gr.sum(axis=1), axis='index')
    loser_density = pred_probs[:, :zero_bin_idx].sum(axis=1)
    gainer_density = pred_probs[:, zero_bin_idx:].sum(axis=1)
    df = pd.DataFrame(
        data=np.vstack((loser_density, gainer_density)).transpose(),
        columns=['Loser Density', 'Gainer Density'])
    df['Actuals'] = acts_ints
    confident_gainer_mask = df['Gainer Density'] >= c_factor * df['Loser Density']
    confident_loser_mask = df['Loser Density'] >= c_factor * df['Gainer Density']
    print(f'shape filtered by confident samples: {df.shape}')
    df = df.loc[confident_gainer_mask | confident_loser_mask]
    print(f'shape filtered by confident samples: {df.shape}')
    gr = df.groupby(by='Actuals').sum()
    gr.columns.name = 'Predictions'
    gl_probabilities = gr.divide(gr.sum(axis=1), axis='index')
    return preds, pred_ints, acts_ints, pred_probabilities, gl_probabilities


def main():
    for s in ('C001', 'C001vn'):
        for t in ('cc', 'oc', 'md'):
            predictions, prediction_ints, actuals_ints, prediction_probs, gl_probs = m057(s, t)
            sn.set(font_scale=0.6)
            ct = pd.crosstab(pd.Series(actuals_ints, name='Actuals'), pd.Series(prediction_ints, name='Predictions'))
            norm_ct = ct / ct.sum(axis=1).values[
                ..., None]  # there is ambiguity btw col names and row names so be careful
            sn.heatmap(ct, annot=True, fmt='d')
            fig = plt.gcf()
            fig.set_size_inches(18, 10.125)
            fig.savefig(f'/home/carl/Downloads/conf_mtrx_using_argmax_{s}_{t}.png', dpi=250)
            fig.clear()
            plt.close()
            sn.heatmap(norm_ct, annot=True, fmt='.1%')
            fig = plt.gcf()
            fig.set_size_inches(18, 10.125)
            fig.savefig(f'/home/carl/Downloads/conf_mtrx_normalized_using_argmax_{s}_{t}.png', dpi=250)
            fig.clear()
            plt.close()
            sn.heatmap(prediction_probs, annot=True, fmt='.1%')
            fig = plt.gcf()
            fig.set_size_inches(18, 10.125)
            fig.savefig(f'/home/carl/Downloads/prediction_prob_dists_{s}_{t}.png', dpi=250)
            fig.clear()
            plt.close()
            sn.heatmap(gl_probs, annot=True, fmt='.1%')
            fig = plt.gcf()
            fig.set_size_inches(4, 10.125)
            fig.savefig(f'/home/carl/Downloads/directionality_prob_dists_{s}_{t}.png', dpi=250)
            fig.clear()
            plt.close()


def yahoo():
    for fname in ('EVAL_Config001_20221015', 'EVAL_Config001VtyNorm_20221015'):
    # for fname in ('CRSP_EVAL_Config001_20221016', 'CRSP_EVAL_Config001VtyNorm_20221016'):
        fpath = f'/mnt/data/trading/datasets/{fname}.hdf5'
        data = InferenceInMemory(fpath)
        for t in ('cc', 'oc', 'md'):
            data.set_target(t)
            s = 'C001' if 'VtyNorm' not in fname else 'C001vn'
            predictions, prediction_ints, actuals_ints, prediction_probs, gl_probs = m057_yahoo(s, t, data)
            sn.set(font_scale=0.6)
            ct = pd.crosstab(pd.Series(actuals_ints, name='Actuals'), pd.Series(prediction_ints, name='Predictions'))
            norm_ct = ct / ct.sum(axis=1).values[
                ..., None]  # there is ambiguity btw col names and row names so be careful
            sn.heatmap(ct, annot=True, fmt='d')
            fig = plt.gcf()
            fig.set_size_inches(18, 10.125)
            fig.savefig(f'/home/carl/Downloads/yahoo_conf_mtrx_using_argmax_{s}_{t}.png', dpi=250)
            fig.clear()
            plt.close()
            sn.heatmap(norm_ct, annot=True, fmt='.1%')
            fig = plt.gcf()
            fig.set_size_inches(18, 10.125)
            fig.savefig(f'/home/carl/Downloads/yahoo_conf_mtrx_normalized_using_argmax_{s}_{t}.png', dpi=250)
            fig.clear()
            plt.close()
            sn.heatmap(prediction_probs, annot=True, fmt='.1%')
            fig = plt.gcf()
            fig.set_size_inches(18, 10.125)
            fig.savefig(f'/home/carl/Downloads/yahoo_prediction_prob_dists_{s}_{t}.png', dpi=250)
            fig.clear()
            plt.close()
            sn.heatmap(gl_probs, annot=True, fmt='.1%')
            fig = plt.gcf()
            fig.set_size_inches(4, 10.125)
            fig.savefig(f'/home/carl/Downloads/yahoo_directionality_prob_dists_{s}_{t}.png', dpi=250)
            fig.clear()
            plt.close()


if __name__ == '__main__':
    # main()
    yahoo()

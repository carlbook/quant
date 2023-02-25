#!/home/carl/miniconda3/envs/trading_pytorch/bin/python

import os
import sys
import h5py
import math
from utils import symmetric_soft_clip
from random import sample
import numpy as np
import pandas as pd

def stock_vec_similarity():
    db_path = '/mnt/data/trading/datasets/CRSPdsf62_trn_1653247514.hdf5'
    rng = np.random.default_rng()
    with h5py.File(db_path, 'r') as db:
        tot_to_read = 1000000
        count_to_analyze = 10000
        vec = db['Fold4']['vec'][:tot_to_read, -1, :]
        idxs = rng.permutation(tot_to_read)
        vec = vec[idxs[:count_to_analyze]]
        dot_prods = []
        for i in range(vec.shape[0]-1):
            for j in range(i+1, vec.shape[0]):
                dot_prods.append(np.dot(vec[i], vec[j]))
        dot_prods = pd.DataFrame(dot_prods)
        print(dot_prods.describe())
        dot_prods.plot.hist(bins=70).get_figure().savefig('/home/carl/Downloads/vector_dotprod_histogram.png')


def super_dot(a1, a2):
    return abs((np.sum(a1 * a2) - a1.shape[0]))


def stock_vec_tsne():
    from sklearn.manifold import TSNE
    db_path = '/mnt/data/trading/datasets/CRSPdsf62_trn_1653247514.hdf5'
    fold = 'Fold4'
    rng = np.random.default_rng()
    with h5py.File(db_path, 'r') as db:
        tot_to_read = 1000000
        count_to_analyze = 10000

        # vec = db[fold]['vec'][:tot_to_read]
        # labels = db[fold]['oc'][:tot_to_read].argmax(axis=1)
        # idxs = rng.permutation(tot_to_read)
        # vec = vec[idxs[:count_to_analyze]]
        # labels = labels[idxs[:count_to_analyze]]
        # num_labels = 20
        # X_2d = TSNE(perplexity=40, metric=super_dot, n_iter=3000, init='pca', learning_rate=200, n_jobs=-1)\
        #     .fit_transform(vec)

        vec = db[fold]['vec'][:tot_to_read, -1, :]
        labels = db[fold]['oc'][:tot_to_read].argmax(axis=1)
        idxs = rng.permutation(tot_to_read)
        vec = vec[idxs[:count_to_analyze]]
        labels = labels[idxs[:count_to_analyze]]
        num_labels = 20
        X_2d = TSNE(perplexity=30, metric=super_dot, n_iter=3000, init='pca', learning_rate=200, n_jobs=-1)\
            .fit_transform(vec)

        from matplotlib import pyplot as plt
        plt.figure(figsize=(12, 9))
        colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
                  '#fabed4', '#469990', '#dcbeff', '#9A6324', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075',
                  '#a9a9a9', '#000000']
        for c, label_num in zip(colors, range(num_labels)):
            plt.scatter(X_2d[labels == label_num, 0], X_2d[labels == label_num, 1], c=c, label=label_num)
        plt.legend()
        plt.savefig(f'/home/carl/Downloads/tsne_{fold}_trn_1653247514_vec_oc.png')


def stock_vec7_optics():
    from sklearn.cluster import OPTICS
    db_path = '/mnt/data/trading/datasets/CRSPdsf62_trn_1653247514.hdf5'
    fold = 'Fold2'
    rng = np.random.default_rng()
    with h5py.File(db_path, 'r') as db:
        tot_to_read = 10000
        vec = db[fold]['vec'][:tot_to_read, -1, :]
        vec = vec[~np.isnan(vec).any(axis=1)]
        idxs = rng.permutation(vec.shape[0])
        vec = vec[idxs]
        clust = OPTICS(min_samples=20)
        clust.fit(vec)
        x = 2


def stock_vec6_optics():
    from sklearn.cluster import OPTICS
    db_path = '/mnt/data/trading/datasets/CRSPdsf62_cleaned_augmented_vectorized_002.hdf5'
    path_out = '/home/carl/trading/quant/docs/data_exploration/'
    rng = np.random.default_rng()
    vectors = []
    tables_to_read = 50
    with h5py.File(db_path, 'r') as db:
        for i, k in enumerate(db.keys()):
            if i % 1000 == 0: print(i)
            if i > tables_to_read: break
            vectors.append(db[k]['Vectors'][...])
    db.close()
    vectors = np.concatenate(vectors)
    vectors = vectors[~np.isnan(vectors).any(axis=1)]
    numel = vectors.shape[0]
    print(numel)
    idxs = rng.permutation(numel)
    vectors = vectors[idxs]
    # distances = []
    # rand_indices = rng.permutation(numel)
    # for i, v in zip(rand_indices, vectors):
    #     distances.append(np.sum(np.abs(v - vectors[i])))
    # distances = np.array(distances)  # trying to figure out what is a reasonable max_eps value
    clust = OPTICS(min_samples=5, min_cluster_size=5000, leaf_size=1000, metric='manhattan', n_jobs=-1, max_eps=0.5)
    labels = clust.fit_predict(vectors)
    x = 2


def stock_vec6_spectral():
    from sklearn.cluster import SpectralClustering as SC
    db_path = '/mnt/data/trading/datasets/CRSPdsf62_cleaned_augmented_vectorized_002.hdf5'
    path_out = '/home/carl/trading/quant/docs/data_exploration/'
    rng = np.random.default_rng()
    vectors = []
    tables_to_read = 30
    with h5py.File(db_path, 'r') as db:
        for i, k in enumerate(db.keys()):
            if i % 1000 == 0: print(i)
            if i > tables_to_read: break
            vectors.append(db[k]['Vectors'][...])
    db.close()
    vectors = np.concatenate(vectors)
    vectors = vectors[~np.isnan(vectors).any(axis=1)]
    numel = vectors.shape[0]
    print(numel)
    idxs = rng.permutation(numel)
    vectors = vectors[idxs]
    clust = SC(n_clusters=100, assign_labels='cluster_qr', n_jobs=-1, affinity='nearest_neighbors', verbose=True)
    labels = clust.fit_predict(vectors)
    x = 2


def stock_vec6_birch():
    from sklearn.cluster import Birch as B, OPTICS
    from matplotlib import pyplot as plt
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    db_path = '/mnt/data/trading/datasets/CRSPdsf62_cleaned_augmented_vectorized_002.hdf5'
    path_out = '/home/carl/trading/quant/docs/data_exploration/'
    rng = np.random.default_rng()
    vectors = []
    # tables_to_read = 3000
    with h5py.File(db_path, 'r') as db:
        for i, k in enumerate(db.keys()):
            if i % 1000 == 0: print(i)
            # if i > tables_to_read: break
            vectors.append(db[k]['Vectors'][...])
    db.close()
    vectors = np.concatenate(vectors)
    vectors = vectors[~np.isnan(vectors).any(axis=1)]
    numel = vectors.shape[0]
    chunksize = numel // 50
    print(numel)
    idxs = rng.permutation(numel)
    vectors = vectors[idxs]
    tr = 0.17
    bf = 100000
    nc = 1000
    # O = OPTICS(min_samples=5, min_cluster_size=0.01, leaf_size=50, metric='manhattan', n_jobs=-1, max_eps=1)
    # clust = B(threshold=tr, branching_factor=bf, n_clusters=O)
    O = None
    clust = B(threshold=tr, branching_factor=bf, n_clusters=nc)
    i = 0
    while i < numel:
        clust.partial_fit(vectors[i: i+chunksize])
        i += chunksize
    labels = clust.predict(vectors)
    df = pd.DataFrame(data=labels, columns=['Cluster'])
    df['Cluster'].value_counts().plot.bar().get_figure().savefig(
        f'{path_out}cln_aug_vec_002_Birch_{tr:.3f}_{bf}_{nc}_{"" if not O else "OPTICS"}.png', dpi=500)
    plt.close()


def stock_vec5_birch():
    from sklearn.cluster import Birch as B, OPTICS
    from matplotlib import pyplot as plt
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    db_path = '/mnt/data/trading/datasets/CRSPdsf62_cleaned_augmented_vectorized_003.hdf5'
    path_out = '/home/carl/trading/quant/docs/data_exploration/'
    rng = np.random.default_rng()
    vectors = []
    # tables_to_read = 3000
    with h5py.File(db_path, 'r') as db:
        for i, k in enumerate(db.keys()):
            if i % 1000 == 0: print(i)
            # if i > tables_to_read: break
            vectors.append(db[k]['Vectors'][...])
    db.close()
    vectors = np.concatenate(vectors)
    vectors = vectors[~np.isnan(vectors).any(axis=1)]
    numel = vectors.shape[0]
    chunksize = numel // 50
    print(numel)
    idxs = rng.permutation(numel)
    vectors = vectors[idxs]
    tr = 0.10
    bf = 1000
    nc = 1000
    # O = OPTICS(min_samples=5, min_cluster_size=0.01, leaf_size=50, metric='manhattan', n_jobs=-1, max_eps=1)
    # clust = B(threshold=tr, branching_factor=bf, n_clusters=O)
    O = None
    clust = B(threshold=tr, branching_factor=bf, n_clusters=nc)
    i = 0
    while i < numel:
        clust.partial_fit(vectors[i: i + chunksize])
        i += chunksize
    labels = clust.predict(vectors)
    df = pd.DataFrame(data=labels, columns=['Cluster'])
    df['Cluster'].value_counts().plot.bar().get_figure().savefig(
        f'{path_out}cln_aug_vec_003_Birch_{tr:.3f}_{bf}_{nc}_{"" if not O else "OPTICS"}.png', dpi=500)
    plt.close()


def stock_vec5_chunked():
    from matplotlib import pyplot as plt
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    db_path = '/mnt/data/trading/datasets/CRSPdsf62_cleaned_augmented_vectorized_003.hdf5'
    path_out = '/home/carl/trading/quant/docs/data_exploration/'
    rng = np.random.default_rng()
    vectors = []
    table_lens = []
    # tables_to_read = 300
    with h5py.File(db_path, 'r') as db:
        for i, k in enumerate(db.keys()):
            if i % 1000 == 0: print(i)
            # if i > tables_to_read: break
            data = db[k]['Vectors'][...]
            table_lens.append(data.shape[0])
            vectors.append(data)
    db.close()
    df_tl = pd.DataFrame(data=table_lens, columns=['Lengths'])
    print(df_tl.describe())
    vectors = np.concatenate(vectors)
    vectors = vectors[~np.isnan(vectors).any(axis=1)]
    numel = vectors.shape[0]
    print(f'\n{numel}\n')
    mins = vectors.min(axis=0)
    maxs = vectors.max(axis=0)
    bins_matrix = np.linspace(start=mins, stop=maxs, num=10)
    oom = 1
    labels = np.zeros(vectors.shape[0], dtype=np.int32)
    for i in range(vectors.shape[1]):
        labels += oom * np.searchsorted(bins_matrix[..., i], vectors[..., i], side='left')
        oom *= 10
    df = pd.DataFrame(data=labels, columns=['Gridpoint'])
    dfg = df['Gridpoint'].value_counts().sort_index()
    print(dfg.describe())
    # # apparently the data is quite sparse, so plotting all possibilities is not helpful
    # all_possibilities = np.zeros(oom)
    # all_possibilities[dfg.index] = dfg.values
    # sq = math.ceil(math.sqrt(oom))
    # all_possibilities = np.hstack((all_possibilities, np.zeros(sq**2 - oom)))  # pad with zeros to get square matrix
    # all_possibilities = all_possibilities.reshape((sq, sq))
    # plt.imshow(all_possibilities, cmap='hot')
    # plt.savefig(f'{path_out}cln_aug_vec_003_VectorGridCounts.png', dpi=500)
    # plt.close()
    sq = math.ceil(math.sqrt(dfg.shape[0]))
    counts_array = np.hstack((dfg.values, np.zeros(sq**2 - dfg.shape[0]))).reshape((sq, sq))
    cutoff = 2000
    counts_array[counts_array > cutoff] = cutoff  # reduce the range so i can see stuff better in the image
    plt.imshow(counts_array, cmap='binary')
    plt.savefig(f'{path_out}cln_aug_vec_003_VectorGridCounts.png', dpi=500)
    plt.close()


def stock_vec5_isomap():
    from matplotlib import pyplot as plt
    from sklearn.manifold import Isomap
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    db_path = '/mnt/data/trading/datasets/CRSPdsf62_cleaned_augmented_vectorized_003.hdf5'
    path_out = '/home/carl/trading/quant/docs/data_exploration/'
    vectors = []
    tables_to_read = 20
    with h5py.File(db_path, 'r') as db:
        for i, k in enumerate(sample(list(db.keys()), tables_to_read)):
            if i % 1000 == 0: print(i)
            data = db[k]['Vectors'][...]
            vectors.append(data)
    db.close()
    vectors = np.concatenate(vectors)
    keep = ~np.isnan(vectors).any(axis=1)
    vectors = vectors[keep]
    numel = vectors.shape[0]
    print(f'\n{numel}\n')
    embedding = Isomap(n_components=2, n_neighbors=5, n_jobs=-1)
    xformd = embedding.fit_transform(vectors)
    plt.scatter(xformd[..., 0], xformd[..., 1], 0.05, alpha=0.3, marker=',')
    plt.legend()
    plt.savefig(f'{path_out}cln_aug_vec_003_IsomapEmbedding.png', dpi=500)
    plt.close()


def stock_vec5_linear_discriminant():
    MDY_BINS = np.array([-0.45, 0, 0.45])
    from matplotlib import pyplot as plt
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    db_path = '/mnt/data/trading/datasets/CRSPdsf62_cleaned_augmented_vectorized_003.hdf5'
    path_out = '/home/carl/trading/quant/docs/data_exploration/'
    vectors, labels = [], []
    tables_to_read = 200
    with h5py.File(db_path, 'r') as db:
        for i, k in enumerate(sample(list(db.keys()), tables_to_read)):
            if i % 1000 == 0: print(i)
            data = db[k]['Vectors'][...]
            classes = np.searchsorted(MDY_BINS, db[k]['Train_Multiday_Perf'][...], side='left').astype(np.int8)
            vectors.append(data)
            labels.append(classes)
    db.close()
    vectors = np.concatenate(vectors)
    labels = np.concatenate(labels)
    keep = ~np.isnan(vectors).any(axis=1)
    vectors = vectors[keep]
    labels = labels[keep]
    numel = vectors.shape[0]
    print(f'\n{numel}\n')
    embedding = LinearDiscriminantAnalysis(n_components=2)
    xformd = embedding.fit_transform(vectors, labels)

    # plt.scatter(xformd[..., 0], xformd[..., 1], 0.5)
    # plt.savefig(f'{path_out}cln_aug_vec_003_LinearDiscriminant.png', dpi=500)

    colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8']
    for c, label_num in zip(colors, range(len(colors))):
        subset_mask = labels == label_num
        plt.scatter(xformd[subset_mask, 0], xformd[subset_mask, 1], 0.05, c=c, label=label_num, alpha=0.3, marker=',')
    plt.legend()
    plt.savefig(f'{path_out}cln_aug_vec_003_LinearDiscriminant.png', dpi=500)
    plt.close()


def stock_vec5_MiniBatchKMeans():
    from matplotlib import pyplot as plt
    from sklearn.cluster import MiniBatchKMeans
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    db_path = '/mnt/data/trading/datasets/CRSPdsf62_cleaned_augmented_vectorized_003.hdf5'
    path_out = '/home/carl/trading/quant/docs/data_exploration/'
    vectors = []
    tables_to_read = 200000000
    with h5py.File(db_path, 'r') as db:
        keys_list = list(db.keys())
        for i, k in enumerate(sample(keys_list, min(tables_to_read, len(keys_list)))):
            if i % 1000 == 0: print(i)
            data = db[k]['Vectors'][...]
            vectors.append(data)
    db.close()
    vectors = np.concatenate(vectors)
    keep = ~np.isnan(vectors).any(axis=1)
    vectors = vectors[keep]
    numel = vectors.shape[0]
    print(f'\n{numel}\n')
    nc = 10000
    it = 100
    rr = 0.1
    mn = 50
    embedding = MiniBatchKMeans(
        n_clusters=nc, max_iter=it, n_init=10, batch_size=2**12, reassignment_ratio=rr, max_no_improvement=mn)
    cluster_id = embedding.fit_predict(vectors)
    print(f'{embedding.n_iter_} iterations were performed')
    df = pd.DataFrame(data=cluster_id, columns=['Cluster'])
    df['Cluster'].value_counts().plot.bar().get_figure().savefig(
        f'{path_out}cln_aug_vec_003_MiniBatchKMeans_{nc}_{it}_{rr}_{mn}.png', dpi=500)
    plt.close()


def stock_vec5_hdbscan():
    pass
#     import hdbscan
#     from sklearn.preprocessing import StandardScaler
#     from matplotlib import pyplot as plt
#     plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
#     db_path = '/mnt/data/trading/datasets/CRSPdsf62_cleaned_augmented_vectorized_003.hdf5'
#     path_out = '/home/carl/trading/quant/docs/data_exploration/'
#     vectors = []
#     tables_to_read = 500
#     with h5py.File(db_path, 'r') as db:
#         keys_list = list(db.keys())
#         for i, k in enumerate(sample(keys_list, min(tables_to_read, len(keys_list)))):
#             if i % 1000 == 0: print(i)
#             data = db[k]['Vectors'][...]
#             vectors.append(data)
#     db.close()
#     vectors = np.concatenate(vectors)
#     keep = ~np.isnan(vectors).any(axis=1)
#     vectors = vectors[keep]
#     scaler = StandardScaler()
#     vectors = scaler.fit_transform(vectors)
#     numel = vectors.shape[0]
#     print(f'\n{numel}\n')
#     size = 20
#     clust = hdbscan.HDBSCAN(min_cluster_size=size, core_dist_n_jobs=-1)
#     embedding = clust.fit_predict(vectors)
#     df = pd.DataFrame(data=embedding, columns=['Cluster'])
#     df['Cluster'].value_counts().plot.bar().get_figure().savefig(
#         f'{path_out}cln_aug_vec_003_hdbscan.png', dpi=500)
#     plt.close()


def stock_vec5_GaussianMixture():
    from matplotlib import pyplot as plt
    from sklearn.mixture import GaussianMixture
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    db_path = '/mnt/data/trading/datasets/CRSPdsf62_cleaned_augmented_vectorized_003.hdf5'
    path_out = '/home/carl/trading/quant/docs/data_exploration/'
    vectors = []
    tables_to_read = 200  # even at just 200 tables it ran for >40 mins and i killed it
    with h5py.File(db_path, 'r') as db:
        keys_list = list(db.keys())
        for i, k in enumerate(sample(keys_list, min(tables_to_read, len(keys_list)))):
            if i % 1000 == 0: print(i)
            data = db[k]['Vectors'][...]
            vectors.append(data)
    db.close()
    vectors = np.concatenate(vectors)
    keep = ~np.isnan(vectors).any(axis=1)
    vectors = vectors[keep]
    numel = vectors.shape[0]
    print(f'\n{numel}\n')
    nc = 1000
    cv = 'full'
    embedding = GaussianMixture(n_components=nc, covariance_type=cv, n_init=3)
    cluster_id = embedding.fit_predict(vectors)
    print(f'{embedding.n_iter_} iterations were performed')
    df = pd.DataFrame(data=cluster_id, columns=['Cluster'])
    df['Cluster'].value_counts().plot.bar().get_figure().savefig(
        f'{path_out}cln_aug_vec_003_GaussianMixture_{nc}_{cv}.png', dpi=500)
    plt.close()


def stock_vec6_MiniBatchKMeans():
    from matplotlib import pyplot as plt
    from sklearn.cluster import MiniBatchKMeans
    # plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    db_path = '/mnt/data/trading/datasets/CRSPdsf62_clust_004.hdf5'
    path_out = '/home/carl/trading/quant/docs/data_exploration/'
    with h5py.File(db_path, 'r') as db:
        vectors = db['stock_vectors'][:100000]
    db.close()
    numel = vectors.shape[0]
    print(f'\n{numel}\n')
    nc = 3000
    it = 10
    rr = 0.3
    mn = 50
    embedding = MiniBatchKMeans(
        n_clusters=nc, max_iter=it, n_init=30, batch_size=2**13, reassignment_ratio=rr, max_no_improvement=mn)
    cluster_id = embedding.fit_predict(vectors)
    print(f'{embedding.n_iter_} iterations were performed')
    df = pd.DataFrame(data=cluster_id, columns=['Cluster'])
    df['Cluster'].value_counts().reset_index(drop=True).plot()
    plt.xlabel('Cluster')
    plt.ylabel('Num Samples')
    plt.savefig(f'{path_out}cln_aug_vec_004_MiniBatchKMeans_{nc}_{it}_{rr}_{mn}.png')
    plt.close()


def training_signal_hists():
    from matplotlib import pyplot as plt
    ds = 'C001vn'
    db_path = f'/mnt/data/trading/datasets/CRSPdsf62_cln_aug_vec_{ds}.hdf5'
    path_out = '/home/carl/trading/quant/docs/data_exploration/'
    returns = []
    co_change = []
    bar_size = []
    cl_in_bar = []
    vol_vs_sma50 = []
    sp500ret = []
    train_cc = []
    train_oc = []
    sp500 = []
    train_mday = []
    limit = np.inf
    with h5py.File(db_path, 'r') as db:
        CC_BINS = db.attrs['cc_bins']
        OC_BINS = db.attrs['oc_bins']
        MDY_BINS = db.attrs['md_bins']
        for i, k in enumerate(db.keys()):
            if i % 1000 == 0: print(i)
            if i > limit: break
            df_vec = pd.DataFrame(db[k]['Vectors'][...])
            returns.append(df_vec.loc[:, 0].values)
            co_change.append(df_vec.loc[:, 1].values)
            bar_size.append(df_vec.loc[:, 2].values)
            cl_in_bar.append(df_vec.loc[:, 3].values)
            vol_vs_sma50.append(df_vec.loc[:, 4].values)
            sp500ret.append(df_vec.loc[:, 5].values)
            train_cc.append(db[k]['Train_CC'][...])
            train_oc.append(db[k]['Train_OC'][...])
            sp500.append(db[k]['SP500_CC'][...])
            train_mday.append(db[k]['Train_Multiday_Perf'][...])

    returns = np.concatenate(returns)
    co_change = np.concatenate(co_change)
    bar_size = np.concatenate(bar_size)
    cl_in_bar = np.concatenate(cl_in_bar)
    vol_vs_sma50 = np.concatenate(vol_vs_sma50)
    sp500ret = np.concatenate(sp500ret)
    train_cc = np.concatenate(train_cc)
    train_oc = np.concatenate(train_oc)
    sp500 = np.concatenate(sp500)
    train_mday = np.concatenate(train_mday)
    group = np.vstack((returns, co_change, bar_size, cl_in_bar,
                       vol_vs_sma50, sp500ret, train_cc, train_oc, sp500, train_mday))
    df = pd.DataFrame(
        data=group.transpose(),
        columns=['Return_noDiv', 'CO_Change', 'BarSize', 'Close_In_Bar',
                 'VolvsSMA50', 'SP500_Ret', 'Train_CC', 'Train_OC', 'SP500_CC', 'Train_Multiday_Perf'])

    print(df.shape)
    df = df.loc[~df.isna().any(axis=1)]
    print(df.shape)
    for c in df.columns:
        print(f'\n{c}')
        print(df[c].describe())

    bins = 100
    df['Return_noDiv'].plot.hist(bins=bins)\
        .get_figure().savefig(f'{path_out}cln_aug_vec_{ds}_ReturnNoDiv_hist.png')
    plt.close()
    df['CO_Change'].plot.hist(bins=bins)\
        .get_figure().savefig(f'{path_out}cln_aug_vec_{ds}_CO_Change_hist.png')
    plt.close()
    df['BarSize'].plot.hist(bins=bins)\
        .get_figure().savefig(f'{path_out}cln_aug_vec_{ds}_BarSize_hist.png')
    plt.close()
    df['Close_In_Bar'].plot.hist(bins=bins, xlim=(-0.2, 1.2))\
        .get_figure().savefig(f'{path_out}cln_aug_vec_{ds}_Close_In_Bar.png')
    plt.close()
    df['VolvsSMA50'].plot.hist(bins=bins)\
        .get_figure().savefig(f'{path_out}cln_aug_vec_{ds}_VolvsSMA50_hist.png')
    plt.close()
    df['SP500_Ret'].plot.hist(bins=bins) \
        .get_figure().savefig(f'{path_out}cln_aug_vec_{ds}_SP500_Ret_hist.png')
    plt.close()
    if 'vn' in ds:
        df['Train_CC'].plot.hist(bins=5*bins, xlim=(-7, 7))\
            .get_figure().savefig(f'{path_out}cln_aug_vec_{ds}_TrainCC_hist.png')
        plt.close()
        df['Train_OC'].plot.hist(bins=5*bins, xlim=(-5, 5))\
            .get_figure().savefig(f'{path_out}cln_aug_vec_{ds}_TrainOC_hist.png')
        plt.close()
    else:
        df['Train_CC'].plot.hist(bins=5*bins, xlim=(-.07, .07))\
            .get_figure().savefig(f'{path_out}cln_aug_vec_{ds}_TrainCC_hist.png')
        plt.close()
        df['Train_OC'].plot.hist(bins=5*bins, xlim=(-.05, .05))\
            .get_figure().savefig(f'{path_out}cln_aug_vec_{ds}_TrainOC_hist.png')
        plt.close()
    df['SP500_CC'].plot.hist(bins=5*bins, xlim=(-.05, .05))\
        .get_figure().savefig(f'{path_out}cln_aug_vec_{ds}_InputSP5_hist.png')
    plt.close()
    df['Train_Multiday_Perf'].plot.hist(bins=5*bins, xlim=(-.6, .6), ylim=(0, 5e5))\
        .get_figure().savefig(f'{path_out}cln_aug_vec_{ds}_TrainMday_hist_truncatedVertical.png')
    plt.close()
    df['binned_CC'] = np.searchsorted(CC_BINS, df['Train_CC'].values, side='left').astype(np.int8)
    df['binned_OC'] = np.searchsorted(OC_BINS, df['Train_OC'].values, side='left').astype(np.int8)
    df['binned_M'] = np.searchsorted(MDY_BINS, df['Train_Multiday_Perf'].values, side='left').astype(np.int8)
    df['binned_CC'].value_counts().sort_index().plot.bar().get_figure()\
        .savefig(f'{path_out}cln_aug_vec_{ds}_cc_bin_valcounts.png')
    plt.close()
    df['binned_OC'].value_counts().sort_index().plot.bar().get_figure()\
        .savefig(f'{path_out}cln_aug_vec_{ds}_oc_bin_valcounts.png')
    plt.close()
    df['binned_M'].value_counts().sort_index().plot.bar().get_figure()\
        .savefig(f'{path_out}cln_aug_vec_{ds}_mday_bin_valcounts.png')
    plt.close()


def cl_vs_sma_hists():
    from matplotlib import pyplot as plt
    from configs import Config001
    config, ds = Config001(), 'C001'
    db_path = f'/mnt/data/trading/datasets/CRSPdsf62_cln_aug_vec_{ds}.hdf5'
    path_out = '/home/carl/trading/quant/docs/data_exploration/'
    c50, c100, c200 = [], [], []
    limit = np.inf
    with h5py.File(db_path, 'r') as db:
        for i, k in enumerate(db.keys()):
            if i % 1000 == 0: print(i)
            if i > limit: break
            shp = config.stk_hist_periods
            cl_series = pd.Series(data=db[k]['Close'][...], dtype=np.float32)
            # the `closed` kwarg must NOT be used on the biggest SMA. That way we only lose first stk_hist_periods - 1
            # values, thus maintaining alignment with the behavior of stack_rolling_window
            cl_vs_sma50 = np.log10((cl_series / cl_series.rolling(50).mean().values)[shp - 1:])
            cl_vs_sma100 = np.log10((cl_series / cl_series.rolling(100).mean().values)[shp - 1:])
            cl_vs_sma200 = np.log10((cl_series / cl_series.rolling(200, min_periods=shp).mean().values)[shp - 1:])
            cl_vs_sma50 = symmetric_soft_clip(cl_vs_sma50.values, config.c50_softclip).astype(np.float32)
            cl_vs_sma100 = symmetric_soft_clip(cl_vs_sma100.values, config.c100_softclip).astype(np.float32)
            cl_vs_sma200 = symmetric_soft_clip(cl_vs_sma200.values, config.c200_softclip).astype(np.float32)
            c50.append(cl_vs_sma50)
            c100.append(cl_vs_sma100)
            c200.append(cl_vs_sma200)

    c50 = np.concatenate(c50)
    c100 = np.concatenate(c100)
    c200 = np.concatenate(c200)

    group = np.vstack((c50, c100, c200))
    df = pd.DataFrame(
        data=group.transpose(),
        columns=['cl_vs_sma50', 'cl_vs_sma100', 'cl_vs_sma200'])

    print(df.shape)
    df = df.loc[~df.isna().any(axis=1)]
    print(df.shape)
    for c in df.columns:
        print(f'\n{c}')
        print(df[c].describe())

    bins = 100
    df['cl_vs_sma50'].plot.hist(bins=bins)\
        .get_figure().savefig(f'{path_out}cl_vs_sma50_{ds}.png')
    plt.close()
    df['cl_vs_sma100'].plot.hist(bins=bins)\
        .get_figure().savefig(f'{path_out}cl_vs_sma100_{ds}.png')
    plt.close()
    df['cl_vs_sma200'].plot.hist(bins=bins)\
        .get_figure().savefig(f'{path_out}cl_vs_sma200_{ds}.png')
    plt.close()

# stock_vec_similarity()
# stock_vec_tsne()
# stock_vec_optics()
# make some box plots once you have clusters.
# stock_vec6_optics()
# stock_vec6_spectral()
# stock_vec6_birch()
# stock_vec5_birch()
# stock_vec5_chunked()
# stock_vec5_isomap()
# stock_vec5_linear_discriminant()
# stock_vec5_MiniBatchKMeans()
# stock_vec5_hdbscan()
# stock_vec5_GaussianMixture()
# stock_vec6_MiniBatchKMeans()
# training_signal_hists()
cl_vs_sma_hists()

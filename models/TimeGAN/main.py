# -*- coding: utf-8 -*-
"""
@Time ： 2022/12/22 15:15
@Auth ： 齐奔
@File ：main.py
@IDE ：PyCharm
@Motto：AIE(Bug Is Everywhere)
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyodbc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from ydata_synthetic.preprocessing.timeseries.utils import real_data_loading
from ydata_synthetic.synthesizers.timeseries import TimeGAN
import matplotlib.gridspec as gridspec

parser = argparse.ArgumentParser(description='dataset process')
parser.add_argument('--raw_data_path', type=str, default='./dataset/LOCA(hot).mdb', help='原始数据路径')
parser.add_argument('--seq_len', type=int, default=96, help='单个样本长度')
parser.add_argument('--n_seq', type=int, default=96, help='特征数量')
parser.add_argument('--hidden_dim', type=int, default=50, help='生成器（GRU/LSTM）隐藏单元数')
parser.add_argument('--gamma', type=int, default=1, help='用于判别器loss')
parser.add_argument('--noise_dim', type=int, default=32, help='噪音起始维度')
parser.add_argument('--dim', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=20, help='batch大小')
parser.add_argument('--lr', type=float, default=5e-4, help='学习率大小')
parser.add_argument('--beta_1', type=int, default=0)
parser.add_argument('--beta_2', type=int, default=1)
parser.add_argument('--data_dim', type=int, default=28)
parser.add_argument('--generated_data_save_path', type=str, default='./dataset/generated_dataset/%s', help='生成数据保存路径')
parser.add_argument('--train_mode', type=bool, default=True, help='是否训练模式')
parser.add_argument('--plot_mode', type=bool, default=True, help='是否绘图模式')
parser.add_argument('--evaluation', type=bool, default=True, help='是否评估假数据集')

args = parser.parse_args()


def load_mdb_dataset():
    driver = '{Microsoft Access Driver (*.mdb, *.accdb)}'
    cnxn = pyodbc.connect(f'Driver={driver};DBQ={args.raw_data_path}')
    df = pd.read_sql('SELECT * FROM PlotData', cnxn, index_col='TIME').iloc[:, :96]
    cols = df.columns
    data = real_data_loading(df.values, seq_len=96)
    return cols, data

def load_csv_dataset():
    path = './dataset/real_dataset/FLB'
    class_df = pd.DataFrame()
    for sample in os.listdir(path):
        sample_path = os.path.join(path, sample)
        sample_df = pd.read_csv(sample_path)
        class_df = pd.concat([class_df,sample_df],ignore_index=True)
    class_df = np.asarray(class_df.iloc[:, 1:97]).reshape(-1, 96, 96)
    # data = real_data_loading(class_df.values, seq_len=96)
    return class_df



def train(data):
    gan_args = [args.batch_size, args.lr, args.beta_1, args.beta_2, args.noise_dim, args.data_dim, args.dim]
    synth = TimeGAN(model_parameters=gan_args, hidden_dim=args.hidden_dim, seq_len=args.seq_len, n_seq=args.n_seq,
                    gamma=1)
    synth.train(data, train_steps=500)
    synth.save('./saved_models/%s.pkl'%accident)


def plot():
    fig, axes = plt.subplots(nrows=24, ncols=4, figsize=(20, 20))
    axes = axes.flatten()
    obs = np.random.randint(len(dataset))
    for j, col in enumerate(columns):
        print(j, col)
        df = pd.DataFrame({'Real': dataset[obs][:, j], 'Synthetic': synth_data[obs][:, j]})
        df.plot(ax=axes[j], title=col, secondary_y='Synthetic data', style=['-', '--'])
    fig.tight_layout()
    if not os.path.exists('./img/%s'%accident):
        os.mkdir('./img/%s'%accident)
    plt.savefig(os.path.join('./img/%s'%accident,'comparison_gan_outputs.png'), dpi=200)


def evaluation():
    sample_size = 50
    idx = np.random.permutation(len(dataset))[:sample_size]  # 随机打散，并抽样250个
    real_sample = np.array(dataset)[idx]
    synthetic_sample = np.array(synth_data)[idx]

    real_sample_reduced = real_sample.reshape(-1, args.seq_len)  # 降维:250，96，96f）-->（250*96f， 96）
    synthetic_sample_reduced = synthetic_sample.reshape(-1, args.seq_len)  # 降维:250，96，96f）-->（250*96f， 96）
    print(real_sample_reduced.shape, synthetic_sample_reduced.shape)
    n_components = 2
    pca = PCA(n_components=n_components)
    tsne = TSNE(n_components=n_components, n_iter=500)

    # pca
    pca.fit(real_sample_reduced)
    pca_real = pd.DataFrame(pca.transform(real_sample_reduced))
    pca_synth = pd.DataFrame(pca.transform(synthetic_sample_reduced))

    # tsne
    data_reduced = np.concatenate((real_sample_reduced, synthetic_sample_reduced), axis=0)
    tsne_results = pd.DataFrame(tsne.fit_transform(data_reduced))

    fig = plt.figure(constrained_layout=True, figsize=(20, 10))
    spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
    ax = fig.add_subplot(spec[0, 0])
    ax.set_title('PCA results', fontsize=20, color='red', pad=10)

    # PCA scatter plot
    plt.scatter(pca_real.iloc[:, 0].values, pca_real.iloc[:, 1].values, c='black', alpha=0.2, label='Original')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)

    plt.scatter(pca_synth.iloc[:, 0].values, pca_synth.iloc[:, 1].values, c='red', alpha=0.2, label='Synthetic')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)

    ax.legend()
    ax2 = fig.add_subplot(spec[0, 1])
    ax2.set_title('TSNE results', fontsize=20, color='red', pad=10)

    # TSNE scatter plot
    plt.scatter(tsne_results.iloc[:sample_size*96, 0].values, tsne_results.iloc[:sample_size*96, 1].values, c='black', alpha=0.2,
                label='Original')

    plt.scatter(tsne_results.iloc[sample_size*96:, 0].values, tsne_results.iloc[sample_size*96:, 1].values, c='red', alpha=0.2,
                label='Synthetic')

    ax2.legend()
    fig.suptitle('Validating synthetic vs real data diversity and distributions', fontsize=16, color='grey')
    if not os.path.exists('./img/%s'%accident):
        os.makedirs('./img/%s'%accident)
    plt.savefig(os.path.join('./img/%s'%accident,'synthetic_vs_real_data_diversity_distributions_'+str(sample_size)), dpi=200)

def synth_data_save(accident_name):
    if not os.path.exists(args.generated_data_save_path % accident_name):
        os.mkdir(args.generated_data_save_path % accident_name)
    for i in range(100):
        sample = np.array(synth_data[i]) #获取单个样本
        sample_df = pd.DataFrame(sample)
        sample_df.columns = columns
        sample_df.index = index
        sample_df.to_csv(os.path.join(args.generated_data_save_path % accident_name, str(i) + '.csv'))

if __name__ == '__main__':
    dataset = load_csv_dataset()  # 真实数据
    columns = pd.read_csv('./dataset/real_dataset/FLB/1.csv').columns[1:97]
    index = pd.read_csv('./dataset/real_dataset/FLB/1.csv').index
    for accident in os.listdir('./dataset/real_dataset'):
        if not os.path.exists('./saved_models/%s.pkl'%accident):
            train(dataset)
        # if args.train_mode:
        synth = TimeGAN.load('./saved_models/%s.pkl' % accident)  # 载入模型
        synth_data = synth.sample(len(dataset))  # 生成数据
        synth_data_save(accident)
        if args.plot_mode:
            plot()
        if args.evaluation:
            evaluation()

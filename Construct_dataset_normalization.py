# -*- coding: utf-8 -*-
"""
@Time ： 2022/12/19 16:35
@Auth ： 齐奔
@File ：Construct_dataset.py
@IDE ：PyCharm
@Motto：AIE(Bug Is Everywhere)
"""

from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np


class My_dataset(Dataset):
    def __init__(self, raw_path, fake_dataset=False):
        """
        Args:
            raw_path: 原始数据路径
        """
        self.feature = []
        self.label = []
        accident_name = os.listdir(raw_path)
        accident_sample = [self._process_item(os.path.join(raw_path, accident), accident, fake_dataset) for accident in accident_name]
        self.label = pd.Categorical(self.label).codes
        assert len(self.label) == len(self.feature)
        # self.mean, self.std = self._getSta() #计算数据集均值和方差时使用

    def _minmax_norm(self, df_input):
        """
        Args:
            df_input: 每个样本的df文件
        Returns:归一化的结果
        """
        for i in list(df_input.columns):
            min = np.min(df_input[i])
            max = np.max(df_input[i])
            max_abs = abs(max) if abs(max) > abs(min) else abs(min)
            df_input[i] = df_input[i] / (max_abs + 1e-6)  # 分母加平滑
        df_input = (df_input - 0.5541133587099674) / 0.4908344488171946 #减均值除标准差
        return df_input.values

    def _getSta(self):

        mean = np.mean(self.feature)
        std = np.std(self.feature)
        print(f'mean:{mean},std:{std}')
        return mean,std

    def _process_item(self, accident_path, accident, fake_dataset):
        """

        Args:
            accident_path: 每个事故得路径
            accident: 事故的名字
        Returns:None

        """
        for sample in os.listdir(accident_path):
            self.label.append(accident)
            sample = pd.read_csv(os.path.join(accident_path, sample)).iloc[:96, 1:97]  # 取前960s数据,后面有部分数据多出三列
            if not fake_dataset:
                sample = self._minmax_norm(sample)
            else:
                sample = sample.values
            # sample = list(itertools.chain.from_iterable(sample)) #将数据展开为1维
            self.feature.append(sample)

    def __getitem__(self, item):
        """
        Args:
            item: 样本索引

        Returns:单个样本

        """
        x = self.feature[item]
        y = self.label[item]
        # if args.label2one_hot:
        #     y = np.eye(args.class_number)[y, :]
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        return x, y

    def __len__(self):
        """
        Returns:样本长度
        """
        return len(self.label)

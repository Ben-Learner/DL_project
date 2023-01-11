import argparse
import pandas as pd
import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from models.GAN import Generator, Discriminator,LSTMClassifier
import time
from pyfiglet import Figlet
from PIL import Image
import imageio
import matplotlib.pyplot as plt


torch.manual_seed(1234)

parser = argparse.ArgumentParser(description='dataset generate')
parser.add_argument('--class_number', type=int, default=12, help='数据集类别数')
parser.add_argument('--label2one_hot', type=bool, default=True, help='构造CGAN输入时，需要将标签转为one-hot向量')
parser.add_argument('--raw_path', type=str, default='./dataset/raw_dataset', help='数据集路径')
parser.add_argument('--epoch', type=int, default=300, help='训练epoch数量')
parser.add_argument('--batch_size', type=int, default=16, help='batch尺寸')
parser.add_argument('--saved_models_path', type=str, default='./saved_models/GAN_models/generator_%d.pth',
                    help='模型保存')
parser.add_argument('--generator_mode', type=bool, default=False, help='生成数据集模式')
parser.add_argument('--train_mode', type=bool, default=False, help='模型训练模式')
parser.add_argument('--LSTM_classifier', type=bool, default=True, help='训练评分器')
parser.add_argument('--TGAN_data2img', type=bool, default=False, help='将TimeGAN生成的数据转为图片')

parser.add_argument('--generator_dataset_save_path', type=str, default='./dataset/generated_dataset/%s',
                    help='生成数据集保存路径')
parser.add_argument('--load_model_number', type=int, default=290, help='用于生成假数据集的模型选取')
parser.add_argument('--raw_data_sta', type=bool, default=False, help='是否生成原始数据集标准化结果')
parser.add_argument('--real_dataset_path', type=str, default='./dataset/real_dataset/%s',
                    help='将原始数据集标准化')
parser.add_argument('--real_data2img_path', type=str, default='./dataset/real_dataset_img/%s',
                    help='将原始数据转为图像的保存路径')
parser.add_argument('--generated_data2img_path', type=str, default='./dataset/generated_dataset_img/%s',
                    help='将CGAN生成数据转为图像的保存路径')
parser.add_argument('--TimeGAN_generated_data2img_path', type=str, default='./dataset/TimeGAN_generated_dataset_img/%s',
                    help='将TimeGAN生成数据转为图像的保存路径')
parser.add_argument('--data2img', type=bool, default=True, help='是否转为图像')
args = parser.parse_args()


class My_dataset(Dataset):
    """
    构造标准数据集，也可以选择是否将数据转成图像
    """
    def __init__(self, raw_path):
        """
        Args:
            raw_path: 原始数据路径
        """
        self.feature = []
        self.label = []
        accident_name = os.listdir(raw_path)
        accident_sample = [self._process_item(os.path.join(raw_path, accident), accident) for accident in accident_name]
        self.label = pd.Categorical(self.label).codes
        assert len(self.label) == len(self.feature)

    def _minmax_norm(self, df_input):
        """
        Args:
            df_input: 每个样本的df文件
        Returns:归一化的结果
        """
        for i in list(df_input.columns):
            Max = np.max(df_input[i])
            df_input[i] = df_input[i] / (Max + 1e-6)  # 分母加平滑
        return df_input

    def _process_item(self, accident_path, accident):
        """

        Args:
            accident_path: 每个事故得路径
            accident: 事故的名字
        Returns:None

        """
        i = 0
        for sample in os.listdir(accident_path):
            self.label.append(accident)
            sample0 = pd.read_csv(os.path.join(accident_path, sample)).iloc[:96, 1:97]  # 取前1500s数据,后面有部分数据多出三列
            sample = self._minmax_norm(sample0)
            if args.raw_data_sta:
                sample.index = sample0.index
                sample.columns = sample0.columns
                if args.data2img:
                    if not os.path.exists(args.real_data2img_path % accident):
                        os.makedirs(args.real_data2img_path % accident)
                    sample_img = (255 * sample.values).astype(np.uint8)
                    # print(sample_img)
                    img = Image.fromarray(sample_img)
                    imageio.imsave(os.path.join(args.real_data2img_path%accident, str(i) + '.jpg'),img)
                else:
                    if not os.path.exists(args.real_dataset_path % accident):
                        os.makedirs(args.real_dataset_path % accident)
                    sample.to_csv(os.path.join(args.real_dataset_path%accident, str(i) + '.csv'))
                i += 1
            # sample = list(itertools.chain.from_iterable(sample)) #将数据展开为1维
            self.feature.append(sample.values)

    def __getitem__(self, item):
        """
        Args:
            item: 样本索引

        Returns:单个样本

        """
        x = self.feature[item]
        y = self.label[item]
        if args.label2one_hot:
            y = np.eye(args.class_number)[y, :]
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        return (x, y)

    def __len__(self):
        """
        Returns:样本长度
        """
        return len(self.label)


f = Figlet(font='slant', width=300)  # 定义打印风格
# 初始化模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gen = Generator(args.class_number).to(device)
dis = Discriminator(args.class_number).to(device)

# 定义损失函数
loss_function = torch.nn.BCELoss()

# 定义优化器
g_optim = torch.optim.SGD(gen.parameters(), lr=4e-4)
d_optim = torch.optim.SGD(dis.parameters(), lr=1e-5)

# 定义loss
D_loss = []
G_loss = []


def train():
    """
    训练CGAN
    Returns:None

    """
    # 加载数据集
    accident_data = My_dataset(args.raw_path)

    dataset = DataLoader(accident_data, batch_size=args.batch_size, shuffle=True)
    for epoch in range(args.epoch):
        d_epoch_loss = 0
        g_epoch_loss = 0
        count = len(accident_data)  # 总样本数
        for step, (sample, label) in enumerate(dataset):
            sample = sample.to(device)
            label = label.to(device)
            random_noise = torch.randn(label.shape[0], 100, device=device)

            d_optim.zero_grad()

            # 判别器在真实样本的损失
            sample = sample.unsqueeze(1)
            real_output = dis(label, sample)
            d_real_loss = loss_function(real_output, torch.ones_like(real_output, device=device))
            d_real_loss.backward()

            # 判别器在生成样本的损失
            gen_sample = gen(label, random_noise).detach()
            fake_output = dis(label, gen_sample)  # 假数据detach(),禁止生成器更新
            d_fake_loss = loss_function(fake_output, torch.zeros_like(fake_output, device=device))
            d_fake_loss.backward()

            # 计算判别器总loss
            d_loss = d_fake_loss + d_real_loss
            d_optim.step()

            # 计算生成器loss
            g_optim.zero_grad()
            gen_sample = gen(label, random_noise)
            fake_output = dis(label, gen_sample)
            g_loss = loss_function(fake_output, torch.ones_like(fake_output, device=device))
            g_loss.backward()
            g_optim.step()

            with torch.no_grad():
                d_epoch_loss += d_loss.item() * label.shape[0]
                g_epoch_loss += g_loss.item() * label.shape[0]
        with torch.no_grad():
            d_epoch_loss /= count
            g_epoch_loss /= count
            D_loss.append(d_epoch_loss)
            G_loss.append(g_epoch_loss)

            if epoch % 10 == 0:
                print('Epoch: {}, d_loss: {:.4f}, g_loss: {:.4f}'.format(epoch, d_epoch_loss, g_epoch_loss))
                torch.save(gen, args.saved_models_path % epoch)
    plt.figure()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(range(args.epoch),D_loss,label='D_loss')
    plt.plot(range(args.epoch),G_loss,label='G_loss')
    plt.legend()
    plt.show()




def generate():
    """
    基于训练好的CGAN生成假数据集，也可以选择是否将数据转为图像
    Returns:None

    """
    model = torch.load(args.saved_models_path % args.load_model_number)
    model.to(device)
    model.eval()
    label_name = os.listdir(args.raw_path)
    column_index_df = pd.read_csv('./dataset/raw_dataset/FLB/1.csv').iloc[:96, 1:97] #为后续生成样本赋予行和列名称

    for label, label_str in enumerate(label_name):
        label = torch.eye(args.class_number)[label, :]  # 转为one-hot变量
        label = label.expand(100, args.class_number).to(device)  # 一一种类型扩增加100个
        random_noise = torch.randn(100, 100, device=device)
        fake_sample = model(label, random_noise).squeeze(1).detach().cpu().numpy()
        if args.data2img:
            if not os.path.exists(args.generated_data2img_path % label_str):
                os.mkdir(args.generated_data2img_path % label_str)
            for sample in range(fake_sample.shape[0]):
                df = (255 * pd.DataFrame(fake_sample[sample, :, :]).values).astype(np.uint8)
                print(df)
                imageio.imsave(os.path.join(args.generated_data2img_path % label_str, str(sample) + '.jpg'), df)
        else:
            if not os.path.exists(args.generator_dataset_save_path % label_str):
                os.mkdir(args.generator_dataset_save_path % label_str)
            for sample in range(fake_sample.shape[0]):
                df = pd.DataFrame(fake_sample[sample, :, :])
                df.index = column_index_df.index
                df.columns = column_index_df.columns
                df.to_csv(os.path.join(args.generator_dataset_save_path % label_str, str(sample) + '.csv'))

def process_raw_data():
    """
    利用该函数将真实数据转为图像
    Returns:

    """
    accident_data = My_dataset(args.raw_path)

def data2img(data_path):
    """
    该函数可以将数据转为图像，与前面代码有重复，限于时间原因，未做优化。
    Args:
        data_path: 数据集路径

    Returns:None

    """
    for accident in os.listdir(data_path):
        if not os.path.exists(args.TimeGAN_generated_data2img_path%accident):
            os.mkdir(args.TimeGAN_generated_data2img_path%accident)
            for number, sample_name in enumerate(os.listdir(os.path.join(data_path, accident))):
                sample_path = os.path.join(os.path.join(data_path,accident),sample_name)
                sample = pd.read_csv(sample_path).values
                sample = (255 * sample).astype(np.uint8) #将sample中的值转为像素值
                print(sample)
                imageio.imsave(os.path.join(args.TimeGAN_generated_data2img_path%accident, str(number) + '.jpg'), sample)

class LSTM_dataset(Dataset):
    """
       构造用于评判生成模型质量的数据集
       """

    def __init__(self, real_path, gen_path):
        """
        Args:
            raw_path: 原始数据路径
        """
        self.feature = []
        self.label = []
        accident_name = os.listdir(real_path)
        real_sample = [self._process_item(os.path.join(real_path, accident), accident, 1) for accident in accident_name]
        generated_sample = [self._process_item(os.path.join(gen_path, accident), accident, 0) for accident in accident_name]
        # self.label = pd.Categorical(self.label).codes
        assert len(self.label) == len(self.feature)

    # def _minmax_norm(self, df_input):
    #     """
    #     Args:
    #         df_input: 每个样本的df文件
    #     Returns:归一化的结果
    #     """
    #     for i in list(df_input.columns):
    #         Max = np.max(df_input[i])
    #         df_input[i] = df_input[i] / (Max + 1e-6)  # 分母加平滑
    #     return df_input

    def _process_item(self, accident_path, accident, label):
        """

        Args:
            accident_path: 每个事故得路径
            accident: 事故的名字
        Returns:None

        """
        for sample in os.listdir(accident_path):
            self.label.append(label)
            sample = pd.read_csv(os.path.join(accident_path, sample)).iloc[:,1:97] # 取前1500s数据,后面有部分数据多出三列
            self.feature.append(sample.values)

    def __getitem__(self, item):
        """
        Args:
            item: 样本索引

        Returns:单个样本

        """
        x = self.feature[item]
        y = self.label[item]
        x = x.astype(np.float32)
        # y = y.astype(np.float32)
        return x, y

    def __len__(self):
        """
        Returns:样本长度
        """
        return len(self.label)

def train_lstm():
    model.train()
    print(f'epoch {epoch}')
    for batchidx, (sample, label) in enumerate(train_loader):
        optimizer.zero_grad()
        sample, label = sample.to(device), label.to(device)
        output = model(sample)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if batchidx % 10 == 0:
            print(f'[{batchidx}/{len(train_loader)}] loss:{loss.item()}')

def eval_lstm():
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_num = 0
        for batchidx, (sample, label) in enumerate(valid_loader):
            sample, label = sample.to(device), label.to(device)
            output = model(sample)
            pred = output.argmax(dim=1)
            total_correct += torch.eq(pred, label).detach().float().sum().item()
            total_num += sample.size(0)
        acc = total_correct / total_num
        print(f'epoch:{epoch}, acc:{acc}')
    return acc

def test_lstm():
    lstm_model = torch.load('./saved_models/GAN_models/LSTM_Classifier_%s.pth'%generated_model_name)
    with torch.no_grad():
        total_correct = 0
        total_num = 0
        for batchidx, (sample, label) in enumerate(test_loader):
            sample, label = sample.to(device), label.to(device)
            output = lstm_model(sample)
            pred = output.argmax(dim=1)
            total_correct += torch.eq(pred, label).detach().float().sum().item()
            total_num += sample.size(0)
        test_acc = total_correct / total_num
        print(f'test_acc:{test_acc}')






if __name__ == "__main__":
    print(time.strftime("%Y-%m-%d %X", time.localtime()))
    if args.generator_mode:
        print(f.renderText('generate dataset'))
        generate()
    if args.train_mode:
        print(f.renderText('train CGAN'))
        train()
    if args.raw_data_sta:
        print(f.renderText('Real'))
        process_raw_data()
    if args.TGAN_data2img:
        data2img('.\dataset\generated_dataset_TimeGAN')
    if args.LSTM_classifier:
        print(f.renderText('train LSTM'))
        generated_model_name = 'TimeGAN'
        # 构建数据集
        accident_data = LSTM_dataset('./dataset/real_dataset', './dataset/generated_dataset_' + generated_model_name)
        # 加载数据集
        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset=accident_data,
                                                                                   lengths=[0.8, 0.1, 0.1])
        train_loader, valid_loader, test_loader = DataLoader(train_dataset, batch_size=16, shuffle=True), \
                                                  DataLoader(valid_dataset, batch_size=16, shuffle=True), \
                                                  DataLoader(test_dataset, batch_size=16)
        if os.path.exists('./saved_models/GAN_models/LSTM_Classifier_%s.pth'%generated_model_name):
            test_lstm()
        else:
            model = LSTMClassifier(input_size=96, hidden_size=256).to(device)
            criterion = torch.nn.CrossEntropyLoss().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            best_acc = 0
            for epoch in range(30):
                train_lstm()
                acc = eval_lstm()
                if best_acc < acc:
                    best_acc = acc
                    torch.save(model,'./saved_models/GAN_models/LSTM_Classifier_%s.pth'%generated_model_name)

    print(time.strftime("%Y-%m-%d %X", time.localtime()))

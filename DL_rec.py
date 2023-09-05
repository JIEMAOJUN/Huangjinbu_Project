import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch import nn
from data_process import fetch_data_from_database


start = time.time()
# 字体配置
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 
# 随机种子设定
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
# torch.manual_seed(1)    # reproducible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 数据库导入
results = fetch_data_from_database('localhost','zzw','123456','hjb','dcs2_selpap')
result_value = results['DCS2_SELPAP']
sqldata = np.array(result_value)
sqldata = sqldata.astype(float)


# 编码器
class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
    # 使用双层LSTM
        self.rnn1 = nn.LSTM(
        input_size=n_features,
        hidden_size=self.hidden_dim,
        num_layers=1,
        batch_first=True)
    
        self.rnn2 = nn.LSTM(
        input_size=self.hidden_dim,
        hidden_size=embedding_dim,
        num_layers=1,
        batch_first=True)

    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        return hidden_n.reshape((self.n_features, self.embedding_dim))
# 解码器
class Decoder(nn.Module):

    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features

        self.rnn1 = nn.LSTM(
        input_size=input_dim,
        hidden_size=input_dim,
        num_layers=1,
        batch_first=True)

        self.rnn2 = nn.LSTM(
        input_size=input_dim,
        hidden_size=self.hidden_dim,
        num_layers=1,
        batch_first=True)
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        x = x.repeat(self.seq_len, self.n_features)
        x = x.reshape((self.n_features, self.seq_len, self.input_dim))
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((self.seq_len, self.hidden_dim))

        return self.output_layer(x)
# 打包解码和编码模块
class RecurrentAutoencoder(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 加载模型
# model = torch.load('model width=50.pth') ##GPU版本
model = torch.load('model width=50.pth',map_location='cpu') ##CPU版本
model = model.to(device)

def rec(runningdata,model):

    test_data = []
    win_width = 50
    for i in range(0,(len(runningdata) - win_width)):          
                test_data.append(runningdata[i:(i + win_width)])

    # 异常数据监测的窗口宽度越小，准确度越高

    # 导入数据
    def create_dataset(arr):
        sequences = arr.astype(np.float32).tolist()
        dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
        n_seq, seq_len, n_features = torch.stack(dataset).shape
        return dataset, seq_len, n_features
    # 零均值归一化
    def Z_score_normalize(data):
        mean_val = np.mean(data)
        std_val = np.std(data)
        normalized_data = [(x-mean_val)/std_val for x in data]
        normalized_data = np.array(normalized_data)    
        return normalized_data

    test_df = Z_score_normalize(test_data)
    test_dataset, _, _ = create_dataset(test_df)

    # 测试模型
    def predict(model, dataset):
        predictions, losses = [], []
        criterion = nn.L1Loss(reduction='sum').to(device)
        with torch.no_grad():
            model = model.eval()
            for seq_true in dataset:
                seq_true = seq_true.to(device)
                seq_pred = model(seq_true)

                loss = criterion(seq_pred, seq_true)

                predictions.append(seq_pred.cpu().numpy().flatten())
                losses.append(loss.item())
        return predictions, losses

    # 重构损失计算
    test_predictions, test_losses = predict(model, test_dataset)












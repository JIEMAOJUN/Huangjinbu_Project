import torch
import copy
import numpy as np
import pandas as pd
import seaborn as sns
import pymysql
from pylab import rcParams
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn, optim
from vmdpy import VMD
from data_process import fetch_data_from_database

# 字体配置文件
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 15, 8
# 随机种子
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


# 训练集长度设定
N = 5000
t = np.arange(0,1000,1)
# guass函数异常
u_1 = 200  
sigma2 = 0.2 
y1 = np.multiply(np.power(np.sqrt(2 * np.pi) * sigma2, -1), 
                 np.exp(-np.power(t - u_1, 2) / 2 * sigma2 ** 2))
# 正弦函数异常
t1 = np.linspace(0,200,200)
sin = lambda x,p:np.sin(2*np.pi*x*t1 + p)
y3 = 30*sin(10,0)
y4 = np.concatenate((np.zeros(500),y3,np.zeros(300)),axis=0)
y4 = y4.reshape(1000,)

ser1 = sqldata[:N]
ser2 = sqldata[N:6000]
ser3 = sqldata[N:6000] + y4

# VMD分解与重构
alpha=2250
tau=0 
K=6 
DC=0 
init=1 
tol=1e-7 

u1, _, _ = VMD(ser1, alpha, tau, K, DC, init, tol) 
u2, _, _ = VMD(ser2, alpha, tau, K, DC, init, tol)
u3, _, _ = VMD(ser3, alpha, tau, K, DC, init, tol)
rec1 = np.sum(u1[0:3], axis=0)
rec2 = np.sum(u2[0:3], axis=0)
rec3 = np.sum(u2[0:3], axis=0)


nor_data = []
test_data = []
ano_data = []
win_width = 50

for i in range(0,(len(rec1) - win_width)):          
            nor_data.append(rec1[i:(i + win_width)])
for i in range(0,(len(rec2) - win_width)):          
            test_data.append(rec2[i:(i + win_width)])
for i in range(0,(len(rec3) - win_width)):          
            ano_data.append(rec3[i:(i + win_width)])
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

normal_df = Z_score_normalize(nor_data)
test_df = Z_score_normalize(test_data)
anomaly_df = Z_score_normalize(ano_data)
print(normal_df.shape)
print(anomaly_df.shape)

train_df, val_df = train_test_split(
  normal_df,
  test_size=0.15,
  random_state=RANDOM_SEED)

train_dataset, seq_len, n_features = create_dataset(train_df)
val_dataset, _, _ = create_dataset(val_df)
test_dataset, _, _ = create_dataset(test_df)
anomaly_dataset, _, _ = create_dataset(anomaly_df)

# for i in range(0,len(train_dataset)):
#     plt.plot(train_dataset[i])


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


model = RecurrentAutoencoder(seq_len, n_features, 128)
model = model.to(device)

# 训练模型
def train_model(model, train_dataset, val_dataset, n_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss(reduction='sum').to(device)
    history = dict(train=[], val=[])
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
  
    for epoch in range(1, n_epochs + 1):
        model = model.train()
        train_losses = []
        for seq_true in train_dataset:
            optimizer.zero_grad()

            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)

            loss = criterion(seq_pred, seq_true)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for seq_true in val_dataset:
                seq_true = seq_true.to(device)
                seq_pred = model(seq_true)

                loss = criterion(seq_pred, seq_true)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

    model.load_state_dict(best_model_wts)
    return model.eval(), history


EPOCH = 80
model, history = train_model(
  model, 
  train_dataset, 
  val_dataset, 
  n_epochs=EPOCH
)

# 绘制loss曲线
ax = plt.figure().gca()
ax.plot(history['train'])
ax.plot(history['val']) 
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'])
plt.title('Loss over training epochs')
plt.show()
# 保存模型
MODEL_PATH = 'model width=50.pth'
torch.save(model, MODEL_PATH)






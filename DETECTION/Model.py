from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_scorefrom sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd

#CNN_model
class feature_cnn(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(feature_cnn, self).__init__()
        self.input_dim = input_dim
        self.cnn1 = torch.nn.Conv1d(in_channels=self.input_dim, out_channels=64, kernel_size=3, stride=1,padding=1) # 60 * 60 --> 50 * 50
        self.pool1 = torch.nn.MaxPool1d(kernel_size=3, stride=1,padding=1) # 50 * 50 --> 25 * 25
        self.cnn2 = torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1,padding=1) # 25 * 25 --> 25 * 25
        self.pool2 = torch.nn.MaxPool1d(kernel_size=3, stride=1,padding=1) # 25 * 25 --> 13 * 13
        self.cnn3 = torch.nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1,padding=1) # 13 * 13 --> 13 * 13
        self.pool3 = torch.nn.MaxPool1d(kernel_size=3, stride=1,padding=1) # 13 * 13 --> 7 * 7
        self.fc1 = torch.nn.Linear(256*33, 32)
        self.active = nn.ReLU()
        self.fc2 = torch.nn.Linear(32, output_dim)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.pool1(x)
        x = self.cnn2(x)
        x = self.pool2(x)
        x = self.cnn3(x)
        x = self.pool3(x)
#         print(x.size())
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.active(x)
        x = self.fc2(x)
#         x = self.active(x)
        return x

#RNN_model
class feature_rnn(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(feature_rnn, self).__init__()
#         self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.relu = nn.ReLU()
        self.hidden_dim = 128
#         self.embed_dim = 256
        self.dense1 = nn.Linear(33, 512)
#         self.pool = torch.nn.MaxPool1d(kernel_size=3, stride=1,padding=1)
#         self.cnn = torch.nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, stride=1,padding=1)
        self.rnn = nn.LSTM(512, self.hidden_dim,
                            num_layers=1,batch_first=True)
        # Maps the output of the LSTM into tag space.
        self.hidden2tag1 = nn.Linear(128, 512)
        self.hidden2tag2 = nn.Linear(512, 1024)
        self.hidden2tag3 = nn.Linear(1024, output_dim)
        self.active = nn.ReLU()
        
    def forward(self,x):
        self.hidden = (torch.randn(1, len(x), 128).to(device),torch.randn(1, len(x), 128).to(device))
        x = x.expand(len(x),8,33)
        x = self.dense1(x)
        rnn_out, hn = self.rnn(x,self.hidden)


        rnn_out = torch.mean(rnn_out,1)
        rnn_out = self.hidden2tag1(rnn_out)
        rnn_out = self.active(rnn_out)
        rnn_out = self.hidden2tag2(rnn_out)
        rnn_out = self.active(rnn_out)
        rnn_out = self.hidden2tag3(rnn_out)
        return rnn_out


from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")
class IDS_CNN():
    def __init__(self,model_name):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(1)
        super(IDS_CNN, self).__init__()
        self.CNN = model_name(1,2).to(device)
        
    def fit(self,xm_train,ym_train):
        self.CNN.train()
        epochs = 10
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(1)
        input_dim = 1
        hidden_dim1 = 10
        output_dim = 2
        BATCH_SIZE = 128
        lr = 0.0001
        # lstm_model = feature_lstm(input_dim,hidden_dim1,output_dim)
        # lstm_model.to(device)

        # scheduler = get_linear_schedule_with_warmup(
        #             optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = AdamW(self.CNN.parameters(), lr=lr)
        train_data = list(zip(xm_train,ym_train))
        dataloader = DataLoader(train_data, batch_size=128, shuffle=True)
#         test_data = list(zip(x_test,y_test))
#         test_dataloader = DataLoader(test_data, batch_size=512, shuffle=False)
        for epoch in range(epochs):
            for batch,data in enumerate(dataloader):
                x_data,label = data
                optimizer.zero_grad()
                x_data = torch.tensor(x_data.float()).to(device)
                x_data = x_data.unsqueeze(1)
                predict = self.CNN(x_data)
                loss = loss_fn(predict, torch.tensor(label).to(device))
                loss.backward()
                optimizer.step()

        return self.CNN
    def predict(self,x_test):
        self.CNN.eval()
        x_test = torch.from_numpy(np.array(x_test))
        with torch.no_grad():
            pre_all = np.zeros(shape=len(x_test))
            test_dataloader = DataLoader(x_test, batch_size=128, shuffle=False)
            for batch,data in enumerate(test_dataloader):
                x_data = data
                x_data = torch.tensor(x_data.float()).to(device)
                x_data = x_data.unsqueeze(1)
                predict = self.CNN(x_data)
                pre = predict.max(1).indices.clone()
                pre = pre.cpu().detach().numpy()
                pre_all[batch*128:batch*128+len(pre)] = pre
        return pre_all
    #     scheduler.step()

#lstm 
class IDS_RNN():
    def __init__(self,model_name):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(1)
        super(IDS_RNN, self).__init__()
        self.rnn = model_name(1,2).to(device)
    def fit(self,xm_train,ym_train):
        self.rnn.train()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(1)
        hidden_dim1 = 10
        BATCH_SIZE = 128
        lr = 0.001
        epochs = 10
        # scheduler = get_linear_schedule_with_warmup(
        #             optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = AdamW(self.rnn.parameters(), lr=lr)
        train_data = list(zip(xm_train,ym_train))
        dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
#         test_data = list(zip(x_test,y_test))
#         test_dataloader = DataLoader(test_data, batch_size=512, shuffle=False)
        for epoch in range(epochs):
            for batch,data in enumerate(dataloader):
                x_data,label = data
                optimizer.zero_grad()
                x_data = torch.tensor(x_data.float()).to(device)
                x_data = x_data.unsqueeze(1)
                predict = self.rnn(x_data)
                loss = loss_fn(predict, torch.tensor(label).to(device))
                loss.backward()
                optimizer.step()
                if batch%400 == 0:
                    print("loss",loss)
        return self.rnn
    def predict(self,x_test):
        self.rnn.eval()
        x_test = torch.from_numpy(np.array(x_test))
        with torch.no_grad():
            pre_all = np.zeros(shape=len(x_test))
            test_dataloader = DataLoader(x_test, batch_size=128, shuffle=False)
            for batch,data in enumerate(test_dataloader):
                x_data = data
                x_data = torch.tensor(x_data.float()).to(device)
                x_data = x_data.unsqueeze(1)
                predict = self.rnn(x_data)
                pre = predict.max(1).indices.clone()
                pre = pre.cpu().detach().numpy()
                pre_all[batch*128:batch*128+len(pre)] = pre
        return pre_all



model_dict = {
            "DT":tree.DecisionTreeClassifier(),
             "BYS":BernoulliNB(),
             "RFC":RandomForestRegressor(),
             "SVM":svm.SVC(kernel='rbf',max_iter=-1, C=5.0,gamma=1),
             "LR":LogisticRegression(),
             "KNN":KNeighborsClassifier(),
             "MLP":MLPClassifier(solver='adam', activation='relu',alpha=1e-4,hidden_layer_sizes=(128,256), random_state=1,max_iter=3,verbose=10,learning_rate_init=0.01),
             "CNN":IDS_CNN(feature_cnn),
             "RNN":IDS_RNN(feature_rnn) 
            }
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# y_resampled = y_resampled.values
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(22)


xm_train, xg_train, ym_train, yg_train = train_test_split(X_resampled, y_resampled, test_size=0.5, random_state=66)
x_train,x_test,y_train,y_test = train_test_split(xm_train,ym_train,test_size=0.5,random_state=34)
for model_name in model_dict:
    print(model_name,"is training")
    model_dict[model_name].fit(x_train,y_train)
    y_pre = model_dict[model_name].predict(x_test)
    y_pre = [round(i) for i in y_pre]
    pre = precision_score(y_test,y_pre)
    rec = recall_score(y_test,y_pre)

    print(model_name,":test_score is",pre,rec)

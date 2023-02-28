import random
import torch
import torch.nn as nn
ngpu = 2
epochs = 3
total_steps=epochs*len(train_adv)//256
warmup_steps = int(total_steps*0.25)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
feature_size = 255

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(22)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

#GAN G
class G(nn.Module):
    def __init__(self):
        super(G,self).__init__()

        self.lin=nn.Sequential(
            nn.Linear(feature_size,256),
#             nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(256,512),
#             nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(512,1024),
#             nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(1024,2048),
            
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(2048,1024),
            
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(1024,feature_size)
        )
        self.out=nn.Tanh()

        
    def char_matrix(self,n):
        a = torch.ones([n,feature_size])
#         a[:,0:2]=0
#         a[:,31:]=0
        return a.to(device).float()

    def forward(self,x):
        x1 = self.lin(x)
        out = self.out(x1)
        out = out*self.char_matrix(out.size(0))
#         combined = torch.cat((x1.view(x1.size(0), -1), x2.view(x2.size(0), -1)), dim=1)
#         out = self.combine_process(combined)
        return torch.clamp(out,0,1),out
#         return x*self.char_matrix(x.size(0)),out
        

class D(nn.Module):
    def __init__(self):
        super(D,self).__init__()
        self.ngpu = ngpu
        self.conv1=nn.Sequential(
            nn.Linear(feature_size,512),
            nn.Dropout(0.3),
            nn.LeakyReLU(), #->(16,28,28)
            nn.Linear(512,16),
            nn.Dropout(0.3),
            nn.LeakyReLU(), #->(16,28,28)
            nn.Linear(16,1),
        )
        
        self.out=nn.Sigmoid()
        

    def forward(self,x):
        x=self.conv1(x)
        output=self.out(x)
        return output

g=G()
d=D()
clip_value=1

g.to(device)
d.to(device)

if torch.cuda.device_count() > 1:
#     g = nn.DataParallel(g,device_ids=[0,1,2])
    g.to(device)
#     d = nn.DataParallel(d,device_ids=[0,1,2])

d.to(device)
import torch.autograd as autograd

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.tensor(np.random.random((real_samples.size(0), 1))).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True).float()
    d_interpolates = D(interpolates)
    fake = torch.ones([real_samples.shape[0],1]).requires_grad_(False).to(device).float()
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_model(model_dict,g,d):
    G_model = {}
    best_mal_sim = None
    for model_name in model_dict:
        print(model_name,"is testing!")
        X_test = torch.tensor(np.array(train_adv)[:,1:256])
        train_Y = torch.tensor([])
        train_data = list(zip(torch.tensor(X_test),torch.tensor(train_y)))
        dataloader = DataLoader(train_data, batch_size=32, shuffle=False)
        all_nums = 0
        for batch,data in enumerate(dataloader):
            x_data,label = data
            predict = model_dict[model_name].predict(np.array(x_data))
            train_Y = torch.cat((train_Y,predict))
        y_pre = model_dict[model_name].predict(np.array(X_test))
        y_pre = [round(i) for i in y_pre]
        dict_four = {}
        for i,four_tuple in enumerate(CTU_four_tuple):
            if four_tuple in dict_four:
                dict_four[four_tuple].append(y_pre[i])
            else:
                dict_four[four_tuple] = [y_pre[i]]

        train_Y = []
        train_adv2 = []
        train_name2 = []
        cccccc = {}
        for i in dict_four:
            if len(dict_four[i])==1:
                cccccc[i] = dict_four[i][0]
        for i,name in enumerate(train_name):
            if name[-1] in cccccc:
                train_name2.append(name)
                train_adv2.append(train_adv[i])
                train_Y.append(cccccc[name[-1]])

        train_adv2 = torch.tensor(train_adv2)
        train_adv2 = train_adv2[:,1:256]

        x_min = train_adv2.min(0).values
        x_max = train_adv2.max(0).values
        train_adv2 = (train_adv2-x_min)/(x_max-x_min)
        #change2
    #     x_min = X_test.min(0).values
    #     x_max = X_test.max(0).values
    #     train_adv2 = (X_test-x_min)/(x_max-x_min)

        #change1
    #     train_adv2 = train_adv2[:,1::2]
    #     xg_train = train_adv2
    #     train_Y = torch.tensor(train_Y)

        xg_train,xg_test,yg_train,yg_test = train_test_split(train_adv2, train_Y, test_size=0.5, random_state=66)
    #     X = np.array(X)
        save_batch = len(xg_train)/128//2
    #     X_test = np.array(X_test)
    #     model = Model_dict[model_name]
    #     predict = model.predict(xg_train)
    #     predict = np.array([1 if i >0.5 else 0 for i in predict])
    #     label = np.array(yg_train)
    #     acc = accuracy_score(predict,label)
    #     print(model_name,"acc is: ",acc)
        benign_data = []
        malicious_data = []
        for i in range(len(xg_train)):
            if yg_train[i] == 0:
                benign_data.append(xg_train[i].numpy())
            else:
                malicious_data.append(xg_train[i].numpy())
        benign_data = random.choices(benign_data,k=len(malicious_data))
        print(model_name,"test DR is: ",len(malicious_data)/len(xg_train))
        benign_data = torch.tensor(benign_data).to(device)
        malicious_data = torch.tensor(malicious_data).to(device)
        G_data = list(zip(benign_data,malicious_data))
        G_dataloader = DataLoader(G_data, batch_size=128, shuffle=True)

        Y = torch.ones([len(xg_test)])
        G_test_data = list(zip(xg_test,Y))
        G_test_dataloader = DataLoader(G_test_data, batch_size=1024, shuffle=True)

        lrG = 0.0001
        lrD = 0.0001
        g=G()
        d=D()
        g.to(device)
        d.to(device)

        epochs = 5
        optimizerD = AdamW(d.parameters(), lr=lrD)
        S_data = list(zip(train_adv2,train_Y))
        loss_fn = nn.BCELoss()
        surrogate_data = DataLoader(S_data, batch_size=128, shuffle=True)
        for epoch in range(epochs):
            for batch,data in enumerate(surrogate_data):
                x_Data,label = data[0],data[1]
                x_Data = x_Data.to(device).float()
                y = label.to(device).float().unsqueeze(1)
                optimizerD.zero_grad()
                y_pre = d(x_Data)
                lossD = loss_fn(y_pre,y)
                lossD.backward()
                optimizerD.step()
                if batch % 500 ==0:
                    y_pre = [1 if i>0.5 else 0 for i in y_pre]
                    print(lossD,f1_score(label,y_pre))

        lrD = 1e-4
    #     d=D()
        optimizerG = AdamW(g.parameters(), lr=lrG)
        optimizerD = AdamW(d.parameters(), lr=lrD)
    #     scheduler = get_linear_schedule_with_warmup(
    #                 optimizerG, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        print(model_name,"is generating adversarial samples")
        best_DR = 1
        for epoch in range(epochs):
            for batches,data_ in enumerate(G_dataloader):
        #         realLabel = data_[1]
                benign,malicious = data_[0],data_[1]
                malicious = malicious.float()
    #             randm = torch.randn([len(malicious),64])
                optimizerD.zero_grad()
                fakeData,noise = g(malicious.to(device))
                fakeData = fakeData.detach()
                fakeData = torch.clamp(fakeData, 0, 1)
    #             gradient_penalty = compute_gradient_penalty(d, benign, fakeData)
                #train D
                lossD = torch.mean(d(benign.to(device).float())) - torch.mean(d(fakeData))#+ gradient_penalty
                lossD.backward()
                optimizerD.step()
                for p in d.parameters():
        #             p.requireds_grad=False
                    p.data.clamp_(-clip_value, clip_value)

                #Discrimitor
                for g_num in range(5):
                    fakeData,noise = g(malicious.to(device))
    #                 fakeData = torch.clamp(fakeData, 0, 1)
                    optimizerG.zero_grad()
                    lossG = torch.mean(d(fakeData)) #+ 0.000001*torch.mean(torch.norm(fakeData.cpu()-malicious.cpu(),1,1))
                    lossG.backward()
                    optimizerG.step()
                if batches%save_batch==0 and batches!=0:
                    TP = 0
                    fP = 0
                    BP = 0
                    all_ = 0
                    print("lossG",lossG,"lossD",lossD)
                    ben_sim = torch.tensor([0.0])
                    mal_sim = torch.tensor([0.0])
                    ben_mal_sim = torch.tensor([0.0])
                    for batches,data_ in enumerate(G_test_dataloader):
                        with torch.no_grad():
                            malicious,y_test = data_[0],data_[1]
                            malicious = malicious.float()
                            fakeData,noise = g(malicious.to(device))
                            fakeData = fakeData.detach()

                            label = np.ones(len(fakeData))
                            predict = d(fakeData.to(device)).max(1).indices
                            predict = np.array([1 if i >0.5 else 0 for i in predict])

                            TP += (label == predict).sum()
                            mal_sim += torch.sum(torch.norm(fakeData.cpu()-malicious.cpu(),2,1))

    #                         ben_mal_sim += torch.sum(torch.norm(malicious.cpu()-benign.cpu(),2,1))
                            all_ += len(fakeData)
                    DR = TP/all_
                    print(model_name, "adversarial acc is:", DR)
                    #save best_DR
                    if best_DR < DR:
                        pass
                    else:
                        G_model[model_name] = g
                        best_DR = DR
                        best_mal_sim = mal_sim/all_
                    print(model_name, "adversarial with mal sim is:",mal_sim/all_)
        print(model_name,":the best DR of adversary is",best_DR,"sim is",best_mal_sim)
    return G_model

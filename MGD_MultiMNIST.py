import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import datasets
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import numpy as np
from msgd import MSGD

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(     
            nn.Conv2d(1, 6, 5, 1, 2),   
            nn.ReLU(),                  
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),           
            nn.MaxPool2d(2, 2)   
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        output = F.log_softmax(x,dim=1)
        return output


def train(cfg,alpha_wt):
    
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
    print("device",device)
    trans_set = cfg["trans_set"]
    print( "Transfer parameters:", trans_set )
    print( "Using alpha_wt", alpha_wt )
    
    EPOCH = 100  
    BATCH_SIZE = 256  
    LR = cfg["learning rate"] 

    params = {
        "batch_size": BATCH_SIZE,
        "dataset": "mnist"
    }

    configs = {
        "mnist": {
            "path": 'data/MultiMNIST',
            "all_tasks": ["L", "R"]
        }
    }

    trainloader, train_dst, testloader, test_dst = datasets.get_dataset(params, configs)

    net_l = LeNet().to( device )
    net_r = LeNet().to( device )

    criterion = nn.CrossEntropyLoss()  

    if "msgd" in cfg["optimizer"]:
        optimizer_l = MSGD( net_l.parameters(), lr=LR, momentum=0.9 )
        optimizer_r = MSGD( net_r.parameters(), lr=LR, momentum=0.9 )
    else:
        optimizer_l = optim.SGD( net_l.parameters(), lr=LR, momentum=0.9 )
        optimizer_r = optim.SGD( net_r.parameters(), lr=LR, momentum=0.9 )

    print("Using Optimizer {}".format( optimizer_l ) )
    scheduler_l = StepLR(optimizer_l, step_size=30, gamma=0.5)
    scheduler_r = StepLR(optimizer_r, step_size=30, gamma=0.5)

    # calculate the mixing matrix
    labels_l_w = torch.Tensor()
    labels_r_w = torch.Tensor()
    for i, data in enumerate(trainloader):
        inputs, labels_l, labels_r = data
        labelsl = labels_l.float()
        labelsr = labels_r.float()
        labels_l_w = torch.cat((labels_l_w, labelsl), dim=0)
        labels_r_w = torch.cat((labels_r_w, labelsr), dim=0)
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    euc = nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)
    input1 = labels_l_w.float()
    input2 = labels_r_w.float()
    min_size = min(input1.size(0), input2.size(0))
    output_cos = cos(input1[0:min_size], input2[0:min_size])
    cos_posi = np.where( output_cos > 0, output_cos, 0 )
    output_cos_rlv = cos_posi/(1+cos_posi)
    output_euc = torch.dist(input1[0:min_size], input2[0:min_size], 2)
    wt = alpha_wt*LR* output_cos_rlv
    print("Using wt", wt)


    for epoch in range(EPOCH):
        sum_loss_l = 0.0
        sum_loss_r = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels_l, labels_r = data
            inputs, labels_l, labels_r = inputs.to(device), labels_l.to(device), labels_r.to(device)

            optimizer_l.zero_grad()
            optimizer_r.zero_grad()

            outputs_l = net_l(inputs)
            loss_l = criterion(outputs_l, labels_l)

            outputs_r = net_r(inputs)
            loss_r = criterion(outputs_r, labels_r)

            loss_l.backward()
            loss_r.backward()
 
            para_l = optimizer_l.param_groups[0]['params']
            para_r = optimizer_r.param_groups[0]['params']
            para_l_mix = []
            para_r_mix = []
            for j in range(len(para_l)):
                if j in trans_set:
                    wti = wt
                else:
                    wti = 0
                paral = para_l[j].data
                parar = para_r[j].data
                para_l_mix.append((1 - wti) * paral + wti * parar)
                para_r_mix.append(wti * paral + (1 - wti) * parar)

            if "msgd" in cfg["optimizer"]:
                optimizer_l.step(para_l_mix)
                optimizer_r.step(para_r_mix)
            else:
                optimizer_l.step()
                optimizer_r.step()

        scheduler_l.step()
        scheduler_r.step()

        if (epoch+1)%100 == 0 or (epoch+1) == EPOCH:
            with torch.no_grad():
                correct_l = 0
                total_l = 0
                correct_r = 0
                total_r = 0
                for data in testloader:
                    images, labels_l, labels_r = data
                    images, labels_l, labels_r = images.to(device), labels_l.to(device), labels_r.to(device)
                    outputs_l = net_l(images)
                    outputs_r = net_r(images)
                    _, predicted_l = torch.max(outputs_l.data, 1)
                    total_l += labels_l.size(0)
                    correct_l += (predicted_l == labels_l).sum()
                    _, predicted_r = torch.max(outputs_r.data, 1)
                    total_r += labels_r.size(0)
                    correct_r += (predicted_r == labels_r).sum()
                print('Epoch %d accuracy on left label：%.03f%%' % (epoch + 1, (100 * correct_l.float() / total_l)))
                print('Epoch %d accuracy on right label：%.03f%%' % (epoch + 1, (100 * correct_r.float() / total_r)))

    # torch.save(net.state_dict(), '%s/net_%03d.pth' % (opt.outf, epoch + 1))


if __name__ == "__main__":
    cfg={
        "trans_set": [0,1,2,3],
        "learning rate": 0.001,
        "optimizer": "msgd"
    }

    for i in range(5):
        train(cfg, 1)

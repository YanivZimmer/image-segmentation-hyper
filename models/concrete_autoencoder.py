import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math


class ConcreteEncoder(nn.Module):
    def __init__(self, input_dim, output_dim,device="cuda", start_temp=8.5, min_temp=0.001, alpha=0.999,headstart_idx=None):#was 8.5 190924
        print("start_temp, min_temp, alpha",start_temp, min_temp, alpha)#start_temp=0.5, min_temp=0.01, alpha=0.99998):
        super().__init__()
        self.headstart_idx = headstart_idx
        #self.headstart_idx=[196,  78,  35]
        #self.headstart_idx=[17, 175,   4,  70,  45]
        self.device = device
        self.start_temp = start_temp
        self.min_temp = min_temp
        self.alpha = alpha
        self.noise = 0.15#0.5#0.15#0.15##prev was 0.5
        self.temp = start_temp
        #out of input_dim select size of output_dim (emaple 25->1). multiple the 1 hot with the original data
        self.logits = nn.Parameter(torch.empty(output_dim, input_dim))
        #default was xavier_normal  nn.init.xavier_normal_(self.logits)
        nn.init.xavier_uniform(self.logits)

        new_logits = 0.0001*self.logits.clone().detach()-0.1#-0.25#-0.06#-0.025

        constant = 0.5#0.1+0.025
        if self.headstart_idx is not None:
            # Iterate over the output_dim
            for i, idx in enumerate(self.headstart_idx):
                # Increase the logits at the corresponding index by a constant
                new_logits[i, idx] += constant

        # for i in range(new_logits.shape[0]+1):
        #     for idx in range(math.ceil(new_logits.shape[1]//new_logits.shape[0])+1):
        #         # Increase the logits at the corresponding index by a constant
        #         print(i, idx+i*(math.ceil(new_logits.shape[1]//new_logits.shape[0])))
        #         if idx+i*(math.ceil(new_logits.shape[1]//new_logits.shape[0]))>new_logits.shape[1]:
        #             break
        # print("start",new_logits.shape[0],new_logits.shape[1])
        # div=new_logits.shape[1]//new_logits.shape[0]
        # print("div",div)
        # gate_idx=0
        # band_idx=0
        # while band_idx<new_logits.shape[1]:
        #     print(band_idx, gate_idx)
        #     for d in range(div):
        #         print(band_idx,gate_idx)
        #         new_logits[gate_idx,band_idx] = constant
        #         band_idx+=1
        #         if band_idx==new_logits.shape[1]:
        #             break
        #     gate_idx+=1
        #     #band_idx += 1

        for i in range(new_logits.shape[0]):
            #for idx in range(new_logits.shape[1]//new_logits.shape[0]+1):
            for idx in range(new_logits.shape[1]//new_logits.shape[0]):
                # Increase the logits at the corresponding index by a constant
                print(i, idx+i*(math.ceil(new_logits.shape[1]//new_logits.shape[0])))
                #if idx+i*(math.ceil(new_logits.shape[1]/new_logits.shape[0]))>=new_logits.shape[1]:
                if i>new_logits.shape[0]:
                    print("hello",i,idx+i*(math.ceil(new_logits.shape[1]//new_logits.shape[0])))
                    break
                #new_logits[i,new_logits.shape[1]-1-(idx+i*(math.ceil(new_logits.shape[1]/new_logits.shape[0])))] += constant
                new_logits[i,idx+i*(math.ceil(new_logits.shape[1]//new_logits.shape[0]))] += constant

        #for j in range(new_logits.shape[1]%new_logits.shape[0]):
            #new_logits[new_logits.shape[0]-1,-j]+=constant
            #print(i,-j)
        print(new_logits[0])
        #print(new_logits[1])
        #print(new_logits[2])
        #print(new_logits[3])
        #print(new_logits[4])
        # Assign the new tensor to self.logits
        #cl_mask =  [1 ,2, 2 ,2 ,0 ,1 ,3 ,4 ,3 ,0 ,4 ,4 ,4 ,4 ,0 ,2 ,2 ,2 ,2 ,0 ,1, 2, 2, 2, 2]  nn.Parameter(self.clustered_mask(cl_mask,5))
        self.logits = nn.Parameter(2.5*new_logits)#
        self.regularizer = lambda : 0
        print(self.logits.detach().cpu().numpy())


    def forward(self, X, train=True, X_mask=None, debug=False):
        #torch.seed()
        uniform = torch.rand(self.logits.shape).clamp(min=1e-7)
        gumbel = -torch.log(-torch.log(uniform)).to(self.device)*self.noise#0.15# it was 0.05 before
        self.temp = max([self.temp * self.alpha, self.min_temp])
        #self.noise = min([self.noise / (self.alpha**2),1])
        #if self.temp>1.5:
        #    self.temp *= self.alpha
        #if self.temp>1.0:
        #    self.temp *= self.alpha
        #if self.temp>2.0:
        #    self.temp *= self.alpha
        #if self.temp > 8.0:
        #    self.temp *= self.alpha

        noisy_logits = (self.logits.to(self.device) + gumbel.to(self.device)) / self.temp

        if X_mask is not None:
            X *= X_mask
            logits_mask = X_mask.int() ^ 1
            noisy_logits = noisy_logits.reshape(1, self.logits.shape[0], -1)
            noisy_logits = torch.add(noisy_logits, logits_mask, alpha=-1e7)

        samples = F.softmax(noisy_logits, dim=-1)
        #print("samples",samples.shape,samples)
        discrete_logits = F.one_hot(torch.argmax(self.logits, dim=-1), self.logits.shape[1]).float()
        #print("discrete",discrete_logits.shape,discrete_logits)
        selection = samples if train else discrete_logits
        #print(X.shape,selection.shape,torch.transpose(selection, -1, -2).shape)
        Y = torch.matmul(X.transpose(1,3), torch.transpose(selection, -1, -2)).transpose(1,3)
        
        if debug:
            return X, selection
        #print("Y",Y.shape,Y)
        return Y
    def clustered_mask(self,mask,n_clusters):
        mask=np.array(mask)
        constant=0.1
        new_logits = 0.0001 * self.logits.clone().detach() - 0.025
        for cluster_idx in range(n_clusters):
            cluster_bands = np.where(mask == cluster_idx,True,False)
            print(cluster_bands)
            print(new_logits.shape)
            new_logits[cluster_idx, cluster_bands] += constant
            print(new_logits)
        return new_logits
    def get_gates(self, mode):
        return self.logits.detach().cpu().numpy(),f"temp={self.temp},start={self.start_temp},alpha={self.alpha},noise={self.noise}"

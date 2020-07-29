import os
import torch
from torch import nn, optim


import numpy as np
import srgan
from tcn import TemporalConvNet as TCN

import matplotlib.pyplot as plt

ONEOVERSQRT2PI = 1.0/np.pi/2.0

def LogMixGaussian(parms,target,batch_size,n_component,index=-1):
    # parms: Shape: [batch_size,3*n_component,length]
    # target: Shape: [batch_size,length]  
    # index: -1 if trained per spectrum. If trained per data point, index is the data point index
    EPS = 1e-9 # To avoid numerical problems
    parms = parms.view(batch_size,n_component,3,parms.size()[2]) 
    if index == -1:
        As = parms[:,:,0,:] #[batch_size,ncomponent,length]
        mus = parms[:,:,1,:]
        sigmas = torch.sigmoid(parms[:,:,2,:])
        As = nn.Softmax(dim=1)(As)
        target = target.expand_as(As)
        singleprob = As*ONEOVERSQRT2PI*torch.exp(-0.5*((target-mus)/sigmas)**2)/sigmas
        combinedprob = torch.sum(singleprob,1) + EPS
        logprob = torch.log(combinedprob)
        totallogprob = torch.sum(logprob,1)
    else:
        As = parms[:,:,0,index] #[batch_size,ncomponent]
        mus = parms[:,:,1,index]
        sigmas = torch.sigmoid(parms[:,:,2,:])
        As = nn.Softmax(dim=1)(As)
        target = target[:,:,index]
        target = target.expand_as(As)
        singleprob = As*ONEOVERSQRT2PI*torch.exp(-0.5*((target-mus)/sigmas)**2)/sigmas
        combinedprob = torch.sum(singleprob,1) + EPS
        logprob = torch.log(combinedprob)
        totallogprob = logprob #torch.sum(logprob,1)
    return -torch.mean(totallogprob)
    
    



if __name__ == '__main__':
    # Load Data
    # Spectra without lines in log scale. Normalized to be between [-0.5,0.5].
    #data = np.load('no_lines.npy')-0.5
    #data = np.load('same_length.npy')-0.5

    # Initial spectra in log scale. Normalized to be between [-0.5,0.5].
    data = np.load('middle_res_shifted.npy')-0.5


    # Parameters
    model_file = 'model.pt' # TCN model name
    n_components = 1 # Number of gaussian components 
    n_layer = 10 # Number of TCN layers
    chans = [2,4,8,16,32,32,32,32,32,n_components*3] # Number of channels in each layer
    #chans = [n_components*3]*n_layer

    nsample = 39996 # Training set size
    batch_size = 36 # Batch size for TCN

    train_tcn = True
    per_datapoint = False # Train TCN per data point or per spectrum
    lr_tcn = 1e-6 
    gamma_tcn = 0.5 
    epochs_tcn = 100 # Total number of epochs
    step_size_tcn = 50 # Reduce LR after this number of epochs

    train_generator = False # Use TCN model to train the generator
    epochs_gen = 200 # Total number of epochs
    lr_gen = 1e-3
    gamma_gen = 0.5
    step_size_gen = 50 # Reduce LR after this number of epochs

    test_gen = False # Show the generated spectrum


    # data processing
    train = data[:nsample]
    test = data[nsample:]
    train = torch.from_numpy(train).float()
    train = train.view(nsample//batch_size,batch_size,1,train.size()[1]) #Shape: [batch_num,batch_size,chan_num,spec_length]

    train = train.cuda()

    # data for generator
    high_res = np.load('middle_res_shifted.npy')-0.5
    low_res = np.array([np.average(high_res[i].reshape(-1,32),axis=1) for i in range(len(high_res))])
    train_gen = low_res[:nsample]
    test_gen_low = low_res[nsample:]
    test_gen_high = high_res[nsample:]

    # construct model
    model = TCN(1,chans,kernel_size=5,dropout=0) # 1 input channel
    generator = srgan.Generator(10,5) # 32 times upsampling
 

    # load/train model
    try:
        model.load_state_dict(torch.load(model_file))  
        print('Load Model')
    except:
        print('No Model found')
        train_tcn = True
    try:
        generator.load_state_dict(torch.load('generator.pt'))
        print("Load Generator Model")
    except:
        print("Generator Model Not Found")




    if train_tcn:
        model = model.cuda()
        optimizer_tcn = optim.Adam(model.parameters(),lr=lr_tcn)
        scheduler_tcn = optim.lr_scheduler.StepLR(optimizer_tcn,step_size=step_size_tcn,gamma=gamma_tcn)
        for j in range(epochs_tcn):
            avg_loss = 0
            optimizer_tcn.zero_grad()
            for i in range(nsample//batch_size):
                itrain = train[i] #[batch_size,1,length]
                if per_datapoint:
                    for k in range(itrain.size()[2]):
                        ioutput = model(itrain).cuda() #[batch_size,3*ncomponent,length]
                        loss = LogMixGaussian(ioutput,itrain,batch_size,n_components,index=k)
                        loss.backward()
                        optimizer_tcn.step()
                else:
                    ioutput = model(itrain).cuda() #[batch_size,3*ncomponent,length]
                    loss = LogMixGaussian(ioutput,itrain,batch_size,n_components)
                    loss.backward()
                    optimizer_tcn.step()
                totloss = LogMixGaussian(ioutput,itrain,batch_size,n_components)
                avg_loss += totloss.data
            avg_loss/=(nsample//batch_size)
            scheduler_tcn.step()
            print(j,avg_loss,scheduler_tcn.get_lr())
        torch.save(model.state_dict(), model_file)


    if train_generator:
        generator.cuda()
        model.cuda()
        
        optimizer_gen = optim.Adam(generator.parameters(),lr=lr_gen)
        scheduler_gen = optim.lr_scheduler.StepLR(optimizer_gen,step_size=step_size_gen,gamma=gamma_gen)
        cov = np.cov(train_gen.T)
        invcov = np.linalg.inv(cov)
        invcov = torch.from_numpy(invcov).float().cuda()

        train_gen = torch.from_numpy(train_gen).float().cuda()
        train_gen = train_gen.view(train_gen.size()[0]//batch_size,batch_size,1,train_gen.size()[1])

        for i in range(epochs_gen):
            avg_loss = 0.0
            for j in range(train_gen.size()[0]):
                idata = train_gen[j]
                ifake = generator(idata).cuda()
                ifake = ifake.view(batch_size,1,train_gen.size()[3]*32)
                ifake_low = ifake.view(batch_size,train_gen.size()[3],32)
                ifake_low = torch.mean(ifake_low,2)
                dev = idata.view(batch_size,-1) - ifake_low
                
                optimizer_gen.zero_grad()
                loss = LogMixGaussian(model(ifake).cuda(),ifake,batch_size,n_components)*batch_size
                for k in range(batch_size):
                    temp = 0.5*torch.matmul(dev[k],torch.matmul(invcov,dev[k]))
                    #print(temp)
                    loss += temp
                loss = loss/batch_size
                loss.backward()
                optimizer_gen.step()
                avg_loss += loss.data
            scheduler_gen.step()
            print(avg_loss/train_gen.size()[0])
        torch.save(generator.state_dict(),'generator.pt')
    if test_gen:
        index = np.random.randint(len(test_gen_low))
        low_input = torch.from_numpy(test_gen_low[index]).float().view(1,1,-1)
        high_output = generator(low_input)
        low_input = low_input.detach().numpy().flatten()
        high_output = high_output.detach().numpy().flatten()
        plt.plot((np.array(range(len(low_input)))+0.5)*32,low_input)
        plt.plot(np.array(range(len(high_output))),high_output)
        plt.plot(np.array(range(len(high_output))),test_gen_high[index])
        plt.savefig('compare.png')
        

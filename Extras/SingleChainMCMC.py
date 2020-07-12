import os.path as osp
import math
import copy
import torch
import random
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU
import numpy as np

import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops

target = 0
dim = 32
batch_size=128


class MyTransform(object):
    def __call__(self, data):
        data.y = data.y[:, target]
        return data


class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data


path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False)])
dataset = QM9(path, transform=transform).shuffle()

mean = dataset.data.y.mean(dim=0, keepdim=True)
std = dataset.data.y.std(dim=0, keepdim=True)
dataset.data.y = (dataset.data.y - mean) / std
mean, std = mean[:, target].item(), std[:, target].item()


test_dataset = dataset[:128]
train_dataset = dataset[2000:2128]
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)


class Net(torch.nn.Module):
    def __init__(self, lrate):
        super(Net, self).__init__()
        self.lin0 = torch.nn.Linear(dataset.num_features, dim)

        nn = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lrate)
        self.criterion = torch.nn.MSELoss()

    def forward(self,x,edge_index,edge_attr,batch):
        out = F.relu(self.lin0(x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, edge_index, edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)

    def evaluate_proposal(self, data, w=None):
        if w is not None:
            self.loadparameters(w)
        y_pred = torch.zeros((len(data), batch_size))
        for i, sample in enumerate(data, 0):
            a = copy.deepcopy(self.forward(sample.x,sample.edge_index,sample.edge_attr,sample.batch).detach())
            y_pred[i] = a
        return y_pred

    def langevin_gradient(self, data, w=None):
        if w is not None:
            self.loadparameters(w)
        self.los = 0
        for i, sample in enumerate(data, 0):
            labels = sample.y
            predicted = self.forward(sample.x,sample.edge_index,sample.edge_attr,sample.batch)
            loss = self.criterion(predicted, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.los += copy.deepcopy(loss.item())
        return copy.deepcopy(self.state_dict())

    def getparameters(self, w=None):
        l = np.array([1, 2])
        dic = {}
        if w is None:
            dic = self.state_dict()
        else:
            dic = copy.deepcopy(w)
        for name in sorted(dic.keys()):
            l = np.concatenate((l, np.array(copy.deepcopy(dic[name])).reshape(-1)), axis=None)
        l = l[2:]
        return l

    def dictfromlist(self, param):
        dic = {}
        i = 0
        for name in sorted(self.state_dict().keys()):
            dic[name] = torch.FloatTensor(param[i:i + (self.state_dict()[name]).view(-1).shape[0]]).view(
                self.state_dict()[name].shape)
            i += (self.state_dict()[name]).view(-1).shape[0]
        # self.loadparameters(dic)
        return dic

    def loadparameters(self, param):
        self.load_state_dict(param)

    def addnoiseandcopy(self, mea, std_dev):
        dic = {}
        w = self.state_dict()
        for name in (w.keys()):
            dic[name] = copy.deepcopy(w[name]) + torch.zeros(w[name].size()).normal_(mean=mea, std=std_dev)
        self.loadparameters(dic)
        return dic

class MCMC:
    def __init__(self, samples, use_langevin_gradients, lr, batch_size):
        self.samples = samples
        self.gnn = Net(lr)
        self.traindata = DataLoader(train_dataset, batch_size=128, shuffle=True)
        self.testdata = DataLoader(test_dataset, batch_size=128, shuffle=False)
        self.use_langevin_gradients = use_langevin_gradients
        self.batch_size = batch_size
        self.l_prob=0.5
        self.tau_sq=25

    def rmse(self, pred, actual):
        return np.sqrt(((pred - actual) ** 2).mean())

    def likelihood_func(self, gnn, data, w=None):
        y = torch.zeros((len(data), self.batch_size))
        for i, dat in enumerate(data, 0):
            labels = dat.y
            y[i]=labels
        if w is not None:
            fx = self.gnn.evaluate_proposal(data, w)
        else:
            fx = self.gnn.evaluate_proposal(data)
        y=y.numpy()
        fx=fx.numpy()
        y=y.ravel()
        fx=fx.ravel()
        rmse = self.rmse(fx, y)
        loss = np.sum(-0.5 * np.log(2 * math.pi * self.tau_sq) - 0.5 * np.square(y - fx) / self.tau_sq)
        lhood=np.sum(loss)
        return [lhood, fx, rmse]

    def prior_likelihood(self, sigma_squared, w_list):
        part1 = -1 * ((len(w_list)) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w_list)))
        log_loss = part1 - part2
        return log_loss

    def accuracy(self, data):
        correct = 0
        total = 0
        for dat in data:
            labels = dat.y
            predicted = self.gnn.forward(dat.x,dat.edge_index,dat.edge_attr,dat.batch)
            total += batch_size
            correct += (predicted == labels).sum().item()
        return 100 * correct / total

    def sampler(self):
        samples = self.samples
        gnn = self.gnn
        w = gnn.state_dict()
        w_size = len(gnn.getparameters(w))
        rmse_train = np.zeros(samples)
        rmse_test = np.zeros(samples)
        acc_train = np.zeros(samples)
        acc_test = np.zeros(samples)

        likelihood_proposal_array = np.zeros(samples)
        likelihood_array=np.zeros(samples)
        diff_likelihood_array=np.zeros(samples)
        weight_array=np.zeros(samples)
        weight_array1=np.zeros(samples)
        weight_array2=np.zeros(samples)
        sum_value_array=np.zeros(samples)

        eta = 0
        w_proposal = np.random.randn(w_size)
        w_proposal = gnn.dictfromlist(w_proposal)
        step_w = 0.05
        train = self.traindata  # data_load(data='train')
        test = self.testdata  # data_load(data= 'test')
        sigma_squared = 25
        prior_current = self.prior_likelihood(sigma_squared, gnn.getparameters(w))

        [likelihood, pred_train, rmsetrain] = self.likelihood_func(gnn, train)
        [_, pred_test, rmsetest] = self.likelihood_func(gnn, test)

        y_test = torch.zeros((len(test), self.batch_size))
        for i, dat in enumerate(test, 0):
            labels = dat.y
            y_test[i] = copy.deepcopy(labels)
        y_train = torch.zeros((len(train), self.batch_size))
        for i, dat in enumerate(train, 0):
            labels = dat.y
            y_train[i] = copy.deepcopy(labels)

        num_accepted = 0
        langevin_count = 0
        init_count = 0
        rmse_train[0] = rmsetrain
        rmse_test[0] = rmsetest
        acc_train[0] = self.accuracy(train)
        acc_test[0] = self.accuracy(test)
        likelihood_proposal_array[0] = 0
        likelihood_array[0] = 0
        diff_likelihood_array[0] = 0
        weight_array[0] = 0
        weight_array1[0] = 0
        weight_array2[0] = 0
        sum_value_array[0] = 0

        for i in range(
                samples):  # Begin sampling --------------------------------------------------------------------------

            lx = np.random.uniform(0, 1, 1)
            old_w = gnn.state_dict()

            if (self.use_langevin_gradients is True) and (lx < self.l_prob):
                w_gd = gnn.langevin_gradient(train)  # Eq 8
                w_proposal = gnn.addnoiseandcopy(0, step_w)  # np.random.normal(w_gd, step_w, w_size) # Eq 7
                w_prop_gd = gnn.langevin_gradient(train)
                wc_delta = (gnn.getparameters(w) - gnn.getparameters(w_prop_gd))
                wp_delta = (gnn.getparameters(w_proposal) - gnn.getparameters(w_gd))
                sigma_sq = step_w
                first = -0.5 * np.sum(wc_delta * wc_delta) / sigma_sq  # this is wc_delta.T  *  wc_delta /sigma_sq
                second = -0.5 * np.sum(wp_delta * wp_delta) / sigma_sq
                diff_prop = first - second
                diff_prop = diff_prop
                langevin_count = langevin_count + 1
            else:
                diff_prop = 0
                w_proposal = gnn.addnoiseandcopy(0, step_w)  # np.random.normal(w, step_w, w_size)

            [likelihood_proposal, pred_train, rmsetrain] = self.likelihood_func(gnn, train)
            [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(gnn, test)

            prior_prop = self.prior_likelihood(sigma_squared,
                                               gnn.getparameters(w_proposal))  # takes care of the gradients

            diff_likelihood = likelihood_proposal - likelihood
            # diff_likelihood = diff_likelihood*-1
            diff_prior = prior_prop - prior_current

            likelihood_proposal_array[i] = likelihood_proposal
            likelihood_array[i] = likelihood
            diff_likelihood_array[i] = diff_likelihood

            sum_value = diff_likelihood + diff_prior + diff_prop
            u = np.log(random.uniform(0, 1))

            sum_value_array[i] = sum_value

            if u < sum_value:
                num_accepted = num_accepted + 1
                likelihood = likelihood_proposal
                prior_current = prior_prop
                w = copy.deepcopy(w_proposal)  # rnn.getparameters(w_proposal)
                acc_train1 = self.accuracy(train)
                acc_test1 = self.accuracy(test)
                print(i, rmsetrain, rmsetest, acc_train1, acc_test1, 'accepted')
                rmse_train[i] = rmsetrain
                rmse_test[i] = rmsetest
                acc_train[i,] = acc_train1
                acc_test[i,] = acc_test1

            else:
                w = old_w
                gnn.loadparameters(w)
                acc_train1 = self.accuracy(train)
                acc_test1 = self.accuracy(test)
                print(i, rmsetrain, rmsetest, acc_train1, acc_test1, 'rejected')
                # rmse_train[i] = rmsetrain
                # rmse_test[i] = rmsetest
                # acc_train[i,] = acc_train1
                # acc_test[i,] = acc_test1
                rmse_train[i,] = rmse_train[i - 1,]
                rmse_test[i,] = rmse_test[i - 1,]
                acc_train[i,] = acc_train[i - 1,]
                acc_test[i,] = acc_test[i - 1,]

            ll = gnn.getparameters()
            weight_array[i] = ll[0]
            weight_array1[i] = ll[100]
            weight_array2[i] = ll[50000]

        print((num_accepted * 100 / (samples * 1.0)), '% was Accepted')

        print((langevin_count * 100 / (samples * 1.0)), '% was Langevin')

        return acc_train, acc_test, rmse_train, rmse_test, sum_value_array, weight_array, weight_array1, weight_array2

def main():

    ulg = True
    numSamples=10
    learnr = 0.01
    burnin=0.25

    mcmc = MCMC(numSamples, ulg, learnr, batch_size)  # declare class
    acc_train, acc_test, rmse_train, rmse_test, sva, wa, wa1, wa2 = mcmc.sampler()

if __name__ == "__main__": main()

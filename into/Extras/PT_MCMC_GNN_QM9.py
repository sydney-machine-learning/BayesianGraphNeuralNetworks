import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU, Sigmoid
import copy
import multiprocessing
import os
import os.path as osp
import numpy as np
import math
import random
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
import torch_geometric.transforms as T

from torch_geometric.datasets import QM9
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops

target = 0
dim = 32
batch_size=128

mpl.use('agg')

torch.backends.cudnn.enabled = False
device = 'cpu'

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

path = osp.join(os.path.dirname(os.path.abspath("__file__")), 'data', 'QM9')
transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False)])
dataset = QM9(path, transform=transform).shuffle()

mean = dataset.data.y.mean(dim=0, keepdim=True)
std = dataset.data.y.std(dim=0, keepdim=True)
dataset.data.y = (dataset.data.y - mean) / std
mean, std = mean[:, target].item(), std[:, target].item()

test_dataset = dataset[1281:1921]
train_dataset = dataset[:1280]

# Initialise and parse command-line inputs

parser = argparse.ArgumentParser(description='PT MCMC GNN')
parser.add_argument('-s', '--samples', help='Number of samples', default=800, dest="samples", type=int)
parser.add_argument('-r', '--replicas', help='Number of chains/replicas, best to have one per availble core/cpu',
                    default=4, dest="num_chains", type=int)
parser.add_argument('-lr', '--learning_rate', help='Learning Rate for Model', dest="learning_rate",
                    default=0.001, type=float)
parser.add_argument('-swap', '--swap', help='Swap Ratio', dest="swap_ratio", default=0.005, type=float)
parser.add_argument('-b', '--burn', help='How many samples to discard before determing posteriors', dest="burn_in",
                    default=0.50, type=float)
parser.add_argument('-pt', '--ptsamples', help='Ratio of PT vs straight MCMC samples to run', dest="pt_samples",
                    default=0.50, type=float)
parser.add_argument('-step', '--step_size', help='Step size for proposals (0.02, 0.05, 0.1 etc)', dest="step_size",
                   default=0.005, type=float)
parser.add_argument('-t', '--temperature', help='Demoninator to determine Max Temperature of chains (MT=no.chains*t) ',
                    default=2, dest="mt_val", type=int)  # Junk
args = parser.parse_args()


def f(): raise Exception("Found exit()")

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
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                               factor=0.7, patience=5,
                                                               min_lr=0.00001)

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
            self.optimizer.zero_grad()
            predicted = self.forward(sample.x,sample.edge_index,sample.edge_attr,sample.batch)
            loss = F.mse_loss(predicted, labels)
            loss.backward()
            self.los += loss.item() * sample.num_graphs
            self.optimizer.step()
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

class ptReplica(multiprocessing.Process):
    def __init__(self, use_langevin_gradients, learn_rate, w, minlim_param, maxlim_param, samples, traindata, testdata,
                 burn_in, temperature, swap_interval, path, parameter_queue, main_process, event, batch_size, step_size):
        self.gnn = Net(learn_rate)
        multiprocessing.Process.__init__(self)
        self.processID = temperature
        self.parameter_queue = parameter_queue
        self.signal_main = main_process
        self.event = event
        self.batch_size = batch_size
        self.temperature = temperature
        self.adapttemp = temperature
        self.swap_interval = swap_interval
        self.path = path
        self.burn_in = burn_in
        self.samples = samples
        self.traindata = DataLoader(train_dataset, batch_size=128, shuffle=True)
        self.testdata = DataLoader(test_dataset, batch_size=128, shuffle=False)
        self.w = w
        self.minY = np.zeros((1, 1))
        self.maxY = np.zeros((1, 1))
        self.minlim_param = minlim_param
        self.maxlim_param = maxlim_param
        self.use_langevin_gradients = use_langevin_gradients
        self.sgd_depth = 1  # Keep as 1
        self.learn_rate = learn_rate
        self.l_prob = 0.5  # Ratio of langevin based proposals, higher value leads to more computation time, evaluate for different problems
        self.step_size = step_size
        self.tau_sq = 0.5

    def error(self, loader):
        error=0
        for data in loader:
            error += (self.gnn(data.x,data.edge_index,data.edge_attr,data.batch) * std - data.y * std).abs().sum().item()  # MAE
        return error/ len(loader.dataset)

    def likelihood_func(self, gnn, loader, tau_pro, w=None):
        y = torch.zeros((len(loader), self.batch_size))
        for i, dat in enumerate(loader, 0):
            labels = dat.y
            y[i]=labels
        if w is not None:
            fx = self.gnn.evaluate_proposal(loader, w)
        else:
            fx = self.gnn.evaluate_proposal(loader)
        error = self.error(loader)
        y=y.numpy()
        fx=fx.numpy()
        y=y.ravel()
        fx=fx.ravel()
        loss = np.sum(-0.5 * np.log(2 * math.pi * tau_pro) - 0.5 * np.square(y - fx) / tau_pro)
        lhood=np.sum(loss)
        return [lhood, fx, error]

    def prior_likelihood(self, sigma_squared, w_list):
        part1 = -1 * ((len(w_list)) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w_list)))
        log_loss = part1 - part2
        return log_loss

    def run(self):
        samples = self.samples
        gnn = self.gnn

        # Random Initialisation of weights
        w = gnn.state_dict()
        w_size = len(gnn.getparameters(w))
        step_w = self.step_size

        rmse_train = np.zeros(samples)
        rmse_test = np.zeros(samples)
        acc_train = np.zeros(samples)
        acc_test = np.zeros(samples)
        weight_array = np.zeros(samples)
        weight_array1 = np.zeros(samples)
        weight_array2 = np.zeros(samples)
        weight_array3 = np.zeros(samples)
        weight_array4 = np.zeros(samples)
        sum_value_array = np.zeros(samples)

        pred_train = gnn.evaluate_proposal(self.traindata)
        pred_train = pred_train.numpy()
        pred_train = pred_train.ravel()

        y_train = torch.zeros((len(self.traindata), self.batch_size))
        for i, dat in enumerate(self.traindata, 0):
            y_train[i] = dat.y

        y_train = y_train.numpy()
        y_train = y_train.ravel()

        eta = np.log(np.var(pred_train - y_train))
        tau_pro = np.exp(eta)

        learn_rate = self.learn_rate

        w_proposal = np.random.randn(w_size)
        w_proposal = gnn.dictfromlist(w_proposal)
        train = self.traindata
        test = self.testdata

        sigma_squared = 25
        prior_current = self.prior_likelihood(sigma_squared, gnn.getparameters(w))  # takes care of the gradients

        # Evaluate Likelihoods

        [likelihood, pred_train, rmsetrain] = self.likelihood_func(gnn, train,tau_pro)

        # print("Hi")
        [_, pred_test, rmsetest] = self.likelihood_func(gnn, test,tau_pro)

        # print("Bye")

        # Beginning sampling using MCMC

        num_accepted = 0
        langevin_count = 0
        pt_samples = samples * 0.6  # PT in canonical form with adaptive temp will work till assigned limit
        init_count = 0

        rmse_train[0] = rmsetrain
        rmse_test[0] = rmsetest
        #acc_train[0] = self.accuracy(train)
        #acc_test[0] = self.accuracy(test)

        weight_array[0] = 0
        weight_array1[0] = 0
        weight_array2[0] = 0
        weight_array3[0] = 0
        weight_array4[0] = 0

        sum_value_array[0] = 0

        for i in range(
                samples):  # Begin sampling --------------------------------------------------------------------------

            ratio = ((samples - i) / (samples * 1.0))
            if i < pt_samples:
                self.adapttemp = self.temperature  # T1=T/log(k+1);
            if i == pt_samples and init_count == 0:  # Move to canonical MCMC
                self.adapttemp = 1
                [likelihood, pred_train, rmsetrain] = self.likelihood_func(gnn, train, tau_pro, w)
                [_, pred_test, rmsetest] = self.likelihood_func(gnn, test,tau_pro, w)
                init_count = 1

            lx = np.random.uniform(0, 1, 1)
            old_w = gnn.state_dict()



            if (self.use_langevin_gradients is True) and (lx < self.l_prob):
                w_gd = gnn.langevin_gradient(train)
                w_proposal = gnn.addnoiseandcopy(0, step_w)
                w_prop_gd = gnn.langevin_gradient(train)
                wc_delta = (gnn.getparameters(w) - gnn.getparameters(w_prop_gd))
                wp_delta = (gnn.getparameters(w_proposal) - gnn.getparameters(w_gd))
                sigma_sq = step_w
                first = -0.5 * np.sum(wc_delta * wc_delta) / sigma_sq
                second = -0.5 * np.sum(wp_delta * wp_delta) / sigma_sq
                diff_prop = first - second
                diff_prop = diff_prop / self.adapttemp
                langevin_count = langevin_count + 1
            else:
                diff_prop = 0
                w_proposal = gnn.addnoiseandcopy(0, step_w)

            [likelihood_proposal, pred_train, rmsetrain] = self.likelihood_func(gnn, train, tau_pro)
            [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(gnn, test, tau_pro)

            prior_prop = self.prior_likelihood(sigma_squared, gnn.getparameters(w_proposal))
            diff_likelihood = likelihood_proposal - likelihood
            diff_prior = prior_prop - prior_current

            """
            try:
                mh_prob = min(1, math.exp(diff_likelihood + diff_prior + diff_prop))
            except OverflowError as e:
                mh_prob = 1
            """


            sum_value = diff_likelihood + diff_prior + diff_prop
            sum_value_array[i] = sum_value
            u = np.log(random.uniform(0, 1))

            if u < sum_value:
                num_accepted = num_accepted + 1
                likelihood = likelihood_proposal
                prior_current = prior_prop
                w = copy.deepcopy(w_proposal)  # rnn.getparameters(w_proposal)
                #acc_train1 = self.accuracy(train)
                #acc_test1 = self.accuracy(test)
                print(i, rmsetrain, rmsetest, 'Accepted')
                rmse_train[i] = rmsetrain
                rmse_test[i] = rmsetest
                #acc_train[i,] = acc_train1
                #acc_test[i,] = acc_test1

            else:
                w = old_w
                gnn.loadparameters(w)
                #acc_train1 = self.accuracy(train)
                #acc_test1 = self.accuracy(test)
                print(i, rmsetrain, rmsetest, 'Rejected')
                rmse_train[i,] = rmse_train[i - 1,]
                rmse_test[i,] = rmse_test[i - 1,]
                #acc_train[i,] = acc_train[i - 1,]
                #acc_test[i,] = acc_test[i - 1,]

            ll = gnn.getparameters()
            weight_array[i] = ll[0]
            weight_array1[i] = ll[100]
            weight_array2[i] = ll[50000]
            weight_array3[i] = ll[40000]
            weight_array4[i] = ll[60000]

            if (i + 1) % self.swap_interval == 0:
                param = np.concatenate([np.asarray([gnn.getparameters(w)]).reshape(-1), np.asarray([eta]).reshape(-1),
                                        np.asarray([likelihood]), np.asarray([self.temperature]), np.asarray([i])])
                self.parameter_queue.put(param)
                self.signal_main.set()
                self.event.clear()
                self.event.wait()
                result = self.parameter_queue.get()
                w = gnn.dictfromlist(result[0:w_size])
                eta = result[w_size]

            if i % 100 == 0:
                print(i, rmsetrain, rmsetest, 'Iteration Number and MAE Train & Test')

        param = np.concatenate(
            [np.asarray([gnn.getparameters(w)]).reshape(-1), np.asarray([eta]).reshape(-1), np.asarray([likelihood]),
             np.asarray([self.temperature]), np.asarray([i])])
        # print('SWAPPED PARAM',self.temperature,param)
        # self.parameter_queue.put(param)
        self.signal_main.set()
        # param = np.concatenate([s_pos_w[i-self.surrogate_interval:i,:],lhood_list[i-self.surrogate_interval:i,:]],axis=1)
        # self.surrogate_parameterqueue.put(param)

        print((num_accepted * 100 / (samples * 1.0)), '% was Accepted')
        accept_ratio = num_accepted / (samples * 1.0) * 100

        print((langevin_count * 100 / (samples * 1.0)), '% was Langevin')
        langevin_ratio = langevin_count / (samples * 1.0) * 100

        print('Exiting the Thread', self.temperature)

        file_name = self.path + '/predictions/sum_value_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, sum_value_array, fmt='%1.2f')

        file_name = self.path + '/predictions/weight[0]_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, weight_array, fmt='%1.2f')

        file_name = self.path + '/predictions/weight[100]_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, weight_array1, fmt='%1.2f')

        file_name = self.path + '/predictions/weight[50000]_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, weight_array2, fmt='%1.2f')

        file_name = self.path + '/predictions/weight[40000]_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, weight_array3, fmt='%1.2f')

        file_name = self.path + '/predictions/weight[60000]_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, weight_array4, fmt='%1.2f')

        file_name = self.path + '/predictions/rmse_test_chain_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, rmse_test, fmt='%1.2f')

        file_name = self.path + '/predictions/rmse_train_chain_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, rmse_train, fmt='%1.2f')

        file_name = self.path + '/predictions/accept_percentage' + str(self.temperature) + '.txt'
        with open(file_name, 'w') as f:
            f.write('%d' % accept_ratio)

class ParallelTempering:
    def __init__(self, use_langevin_gradients, learn_rate, num_chains, maxtemp, NumSample, swap_interval,
                 path, batch_size, bi, step_size):
        gnn = Net(learn_rate)
        self.gnn = gnn
        self.traindata = DataLoader(train_dataset, batch_size=128, shuffle=True)
        self.testdata = DataLoader(test_dataset, batch_size=128, shuffle=False)
        self.num_param = len(gnn.getparameters(
            gnn.state_dict()))  # (topology[0] * topology[1]) + (topology[1] * topology[2]) + topology[1] + topology[2]
        # Parallel Tempering variables
        self.swap_interval = swap_interval
        self.path = path
        self.maxtemp = maxtemp
        self.num_swap = 0
        self.total_swap_proposals = 0
        self.num_chains = num_chains
        self.chains = []
        self.temperatures = []
        self.NumSamples = int(NumSample / self.num_chains)
        self.sub_sample_size = max(1, int(0.05 * self.NumSamples))
        # create queues for transfer of parameters between process chain
        self.parameter_queue = [multiprocessing.Queue() for i in range(num_chains)]
        self.chain_queue = multiprocessing.JoinableQueue()
        self.wait_chain = [multiprocessing.Event() for i in range(self.num_chains)]
        self.event = [multiprocessing.Event() for i in range(self.num_chains)]
        self.all_param = None
        self.geometric = True  # True (geometric)  False (Linear)
        self.minlim_param = 0.0
        self.maxlim_param = 0.0
        self.minY = np.zeros((1, 1))
        self.maxY = np.ones((1, 1))
        self.model_signature = 0.0
        self.learn_rate = learn_rate
        self.use_langevin_gradients = use_langevin_gradients
        self.batch_size = batch_size
        self.masternumsample = NumSample
        self.burni = bi
        self.step_size = step_size

    def default_beta_ladder(self, ndim, ntemps,
                            Tmax):  # https://github.com/konqr/ptemcee/blob/master/ptemcee/sampler.py
        """
        Returns a ladder of :math:`\beta \equiv 1/T` under a geometric spacing that is determined by the
        arguments ``ntemps`` and ``Tmax``.  The temperature selection algorithm works as follows:
        Ideally, ``Tmax`` should be specified such that the tempered posterior looks like the prior at
        this temperature.  If using adaptive parallel tempering, per `arXiv:1501.05823
        <http://arxiv.org/abs/1501.05823>`_, choosing ``Tmax = inf`` is a safe bet, so long as
        ``ntemps`` is also specified.
        """
        if type(ndim) != int or ndim < 1:
            raise ValueError('Invalid number of dimensions specified.')
        if ntemps is None and Tmax is None:
            raise ValueError('Must specify one of ``ntemps`` and ``Tmax``.')
        if Tmax is not None and Tmax <= 1:
            raise ValueError('``Tmax`` must be greater than 1.')
        if ntemps is not None and (type(ntemps) != int or ntemps < 1):
            raise ValueError('Invalid number of temperatures specified.')

        tstep = np.array([25.2741, 7., 4.47502, 3.5236, 3.0232,
                          2.71225, 2.49879, 2.34226, 2.22198, 2.12628,
                          2.04807, 1.98276, 1.92728, 1.87946, 1.83774,
                          1.80096, 1.76826, 1.73895, 1.7125, 1.68849,
                          1.66657, 1.64647, 1.62795, 1.61083, 1.59494,
                          1.58014, 1.56632, 1.55338, 1.54123, 1.5298,
                          1.51901, 1.50881, 1.49916, 1.49, 1.4813,
                          1.47302, 1.46512, 1.45759, 1.45039, 1.4435,
                          1.4369, 1.43056, 1.42448, 1.41864, 1.41302,
                          1.40761, 1.40239, 1.39736, 1.3925, 1.38781,
                          1.38327, 1.37888, 1.37463, 1.37051, 1.36652,
                          1.36265, 1.35889, 1.35524, 1.3517, 1.34825,
                          1.3449, 1.34164, 1.33847, 1.33538, 1.33236,
                          1.32943, 1.32656, 1.32377, 1.32104, 1.31838,
                          1.31578, 1.31325, 1.31076, 1.30834, 1.30596,
                          1.30364, 1.30137, 1.29915, 1.29697, 1.29484,
                          1.29275, 1.29071, 1.2887, 1.28673, 1.2848,
                          1.28291, 1.28106, 1.27923, 1.27745, 1.27569,
                          1.27397, 1.27227, 1.27061, 1.26898, 1.26737,
                          1.26579, 1.26424, 1.26271, 1.26121,
                          1.25973])

        if ndim > tstep.shape[0]:
            # An approximation to the temperature step at large
            # dimension
            tstep = 1.0 + 2.0 * np.sqrt(np.log(4.0)) / np.sqrt(ndim)
        else:
            tstep = tstep[ndim - 1]

        appendInf = False
        if Tmax == np.inf:
            appendInf = True
            Tmax = None
            ntemps = ntemps - 1

        if ntemps is not None:
            if Tmax is None:
                # Determine Tmax from ntemps.
                Tmax = tstep ** (ntemps - 1)
        else:
            if Tmax is None:
                raise ValueError('Must specify at least one of ``ntemps'' and '
                                 'finite ``Tmax``.')

            # Determine ntemps from Tmax.
            ntemps = int(np.log(Tmax) / np.log(tstep) + 2)

        betas = np.logspace(0, -np.log10(Tmax), ntemps)
        if appendInf:
            # Use a geometric spacing, but replace the top-most temperature with
            # infinity.
            betas = np.concatenate((betas, [0]))

        return betas

    def assign_temperatures(self):
        if self.geometric == True:
            betas = self.default_beta_ladder(2, ntemps=self.num_chains, Tmax=self.maxtemp)
            for i in range(0, self.num_chains):
                self.temperatures.append(np.inf if betas[i] == 0 else 1.0 / betas[i])
                # print (self.temperatures[i])
        else:

            tmpr_rate = (self.maxtemp / self.num_chains)
            temp = 1
            for i in range(0, self.num_chains):
                self.temperatures.append(temp)
                temp += tmpr_rate
                # print(self.temperatures[i])

    def initialize_chains(self, burn_in):
        self.burn_in = burn_in
        self.assign_temperatures()
        self.minlim_param = np.repeat([-100], self.num_param)  # priors for nn weights
        self.maxlim_param = np.repeat([100], self.num_param)
        for i in range(0, self.num_chains):
            w = np.random.randn(self.num_param)
            w = self.gnn.dictfromlist(w)
            self.chains.append(
                ptReplica(self.use_langevin_gradients, self.learn_rate, w, self.minlim_param, self.maxlim_param,
                          self.NumSamples, self.traindata, self.testdata, self.burn_in,
                          self.temperatures[i], self.swap_interval, self.path, self.parameter_queue[i],
                          self.wait_chain[i], self.event[i], self.batch_size, self.step_size))

    def surr_procedure(self, queue):
        if queue.empty() is False:
            return queue.get()
        else:
            return

    def swap_procedure(self, parameter_queue_1, parameter_queue_2):
        #        if parameter_queue_2.empty() is False and parameter_queue_1.empty() is False:
        param1 = parameter_queue_1.get()
        param2 = parameter_queue_2.get()
        w1 = param1[0:self.num_param]
        eta1 = param1[self.num_param]
        lhood1 = param1[self.num_param + 1]
        T1 = param1[self.num_param + 2]
        w2 = param2[0:self.num_param]
        eta2 = param2[self.num_param]
        lhood2 = param2[self.num_param + 1]
        T2 = param2[self.num_param + 2]
        # print('yo')
        # SWAPPING PROBABILITIES
        try:
            swap_proposal = min(1, 0.5 * np.exp(lhood2 - lhood1))
        except OverflowError:
            swap_proposal = 1
        u = np.random.uniform(0, 1)
        if u < swap_proposal:
            swapped = True
            self.total_swap_proposals += 1
            self.num_swap += 1
            param_temp = param1
            param1 = param2
            param2 = param_temp
        else:
            swapped = False
            self.total_swap_proposals += 1
        return param1, param2, swapped

    def run_chains(self):
        # only adjacent chains can be swapped therefore, the number of proposals is ONE less num_chains
        # swap_proposal = np.ones(self.num_chains-1)
        # create parameter holders for paramaters that will be swapped
        # replica_param = np.zeros((self.num_chains, self.num_param))
        # lhood = np.zeros(self.num_chains)
        # Define the starting and ending of MCMC Chains
        start = 0
        end = self.NumSamples - 1
        # number_exchange = np.zeros(self.num_chains)
        # filen = open(self.path + '/num_exchange.txt', 'a')
        # RUN MCMC CHAINS
        for l in range(0, self.num_chains):
            self.chains[l].start_chain = start
            self.chains[l].end = end
        for j in range(0, self.num_chains):
            self.wait_chain[j].clear()
            self.event[j].clear()
            self.chains[j].start()
        # SWAP PROCEDURE
        swaps_affected_main = 0
        total_swaps = 0
        for i in range(int(self.NumSamples / self.swap_interval)):
            # print(i,int(self.NumSamples/self.swap_interval), 'Counting')
            count = 0
            for index in range(self.num_chains):
                if not self.chains[index].is_alive():
                    count += 1
                    self.wait_chain[index].set()
                    # print(str(self.chains[index].temperature) + " Dead" + str(index))

            if count == self.num_chains:
                break
            # print(count,'Is the Count')
            timeout_count = 0
            for index in range(0, self.num_chains):
                # print("Waiting for chain: {}".format(index+1))
                flag = self.wait_chain[index].wait()
                if flag:
                    # print("Signal from chain: {}".format(index+1))
                    timeout_count += 1

            if timeout_count != self.num_chains:
                # print("Skipping the Swap!")
                continue
            # print("Event Occured")
            for index in range(0, self.num_chains - 1):
                # print('Starting Swap')
                swapped = False
                param_1, param_2, swapped = self.swap_procedure(self.parameter_queue[index],
                                                                self.parameter_queue[index + 1])
                self.parameter_queue[index].put(param_1)
                self.parameter_queue[index + 1].put(param_2)
                if index == 0:
                    if swapped:
                        swaps_affected_main += 1
                    total_swaps += 1
            for index in range(self.num_chains):
                self.wait_chain[index].clear()
                self.event[index].set()

        print("Joining Processes")

        # JOIN THEM TO MAIN PROCESS
        for index in range(0, self.num_chains):
            print('Waiting to Join ', index, self.num_chains)
            print(self.chains[index].is_alive())
            self.chains[index].join()
            print(index, 'Chain Joined')
        self.chain_queue.join()
        # pos_w, fx_train, fx_test, rmse_train, rmse_test, acc_train, acc_test, likelihood_vec, accept_vec, accept = self.show_results()
        rmse_train, rmse_test, apal = self.show_results()
        print("NUMBER OF SWAPS = ", self.num_swap)
        swap_perc = self.num_swap * 100 / self.total_swap_proposals
        # return pos_w, fx_train, fx_test, rmse_train, rmse_test, acc_train, acc_test, likelihood_vec, swap_perc, accept_vec, accept
        return rmse_train, rmse_test, apal, swap_perc

    def show_results(self):
        burnin = int(self.NumSamples * self.burn_in)
        mcmc_samples = int(self.NumSamples * 0.25)
        # likelihood_rep = np.zeros((self.num_chains, self.NumSamples - burnin,2))  # index 1 for likelihood posterior and index 0 for Likelihood proposals. Note all likilihood proposals plotted only
        # accept_percent = np.zeros((self.num_chains, 1))
        # accept_list = np.zeros((self.num_chains, self.NumSamples))
        # pos_w = np.zeros((self.num_chains, self.NumSamples - burnin, self.num_param))
        # fx_train_all = np.zeros((self.num_chains, self.NumSamples - burnin, len(self.traindata)))
        rmse_train = np.zeros((self.num_chains, self.NumSamples))
        #acc_train = np.zeros((self.num_chains, self.NumSamples))

        # fx_test_all = np.zeros((self.num_chains, self.NumSamples - burnin, len(self.testdata)))
        rmse_test = np.zeros((self.num_chains, self.NumSamples))
        #acc_test = np.zeros((self.num_chains, self.NumSamples))
        sum_val_array = np.zeros((self.num_chains, self.NumSamples))

        weight_ar = np.zeros((self.num_chains, self.NumSamples))
        weight_ar1 = np.zeros((self.num_chains, self.NumSamples))
        weight_ar2 = np.zeros((self.num_chains, self.NumSamples))
        weight_ar3 = np.zeros((self.num_chains, self.NumSamples))
        weight_ar4 = np.zeros((self.num_chains, self.NumSamples))

        accept_percentage_all_chains = np.zeros(self.num_chains)

        for i in range(self.num_chains):
            # file_name = self.path + '/posterior/pos_w/' + 'chain_' + str(self.temperatures[i]) + '.txt'
            # print(self.path)
            # print(file_name)
            # dat = np.loadtxt(file_name)
            # pos_w[i, :, :] = dat[burnin:, :]

            # file_name = self.path + '/posterior/pos_likelihood/' + 'chain_' + str(self.temperatures[i]) + '.txt'
            # dat = np.loadtxt(file_name)
            # likelihood_rep[i, :] = dat[burnin:]

            # file_name = self.path + '/posterior/accept_list/' + 'chain_' + str(self.temperatures[i]) + '.txt'
            # dat = np.loadtxt(file_name)
            # accept_list[i, :] = dat

            file_name = self.path + '/predictions/rmse_test_chain_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            rmse_test[i, :] = dat

            file_name = self.path + '/predictions/rmse_train_chain_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            rmse_train[i, :] = dat

            #file_name = self.path + '/predictions/acc_test_chain_' + str(self.temperatures[i]) + '.txt'
            #dat = np.loadtxt(file_name)
            #acc_test[i, :] = dat

            #file_name = self.path + '/predictions/acc_train_chain_' + str(self.temperatures[i]) + '.txt'
            #dat = np.loadtxt(file_name)
            #acc_train[i, :] = dat

            file_name = self.path + '/predictions/sum_value_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            sum_val_array[i, :] = dat

            file_name = self.path + '/predictions/weight[0]_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            weight_ar[i, :] = dat

            file_name = self.path + '/predictions/weight[100]_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            weight_ar1[i, :] = dat

            file_name = self.path + '/predictions/weight[50000]_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            weight_ar2[i, :] = dat

            file_name = self.path + '/predictions/weight[40000]_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            weight_ar3[i, :] = dat

            file_name = self.path + '/predictions/weight[60000]_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            weight_ar4[i, :] = dat

            file_name = self.path + '/predictions/accept_percentage' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            accept_percentage_all_chains[i] = dat

        rmse_train_single_chain_plot = rmse_train[0, :]
        rmse_test_single_chain_plot = rmse_test[0, :]
        #acc_train_single_chain_plot = acc_train[0, :]
        #acc_test_single_chain_plot = acc_test[0, :]
        sum_val_array_single_chain_plot = sum_val_array[0]

        path = 'GNN/graphs'

        x2 = np.linspace(0, self.NumSamples, num=self.NumSamples)

        plt.plot(x2, sum_val_array_single_chain_plot, label='Sum Value')
        plt.legend(loc='upper right')
        plt.title("Sum Value Single Chain")
        plt.savefig(path + '/sum_value_single_chain.png')
        plt.clf()

        color = 'tab:red'
        plt.plot(x2, rmse_train_single_chain_plot, label="Train", color=color)
        color = 'tab:blue'
        plt.plot(x2, rmse_test_single_chain_plot, label="Test", color=color)
        plt.xlabel('Samples')
        plt.ylabel('RMSE')
        plt.legend()
        plt.savefig(path + '/superimposed_rmse_single_chain.png')
        plt.clf()

        rmse_train = rmse_train.reshape((self.num_chains * self.NumSamples), 1)
        #acc_train = acc_train.reshape((self.num_chains * self.NumSamples), 1)
        rmse_test = rmse_test.reshape((self.num_chains * self.NumSamples), 1)
        #acc_test = acc_test.reshape((self.num_chains * self.NumSamples), 1)
        sum_val_array = sum_val_array.reshape((self.num_chains * self.NumSamples), 1)
        weight_ar = weight_ar.reshape((self.num_chains * self.NumSamples), 1)
        weight_ar1 = weight_ar1.reshape((self.num_chains * self.NumSamples), 1)
        weight_ar2 = weight_ar2.reshape((self.num_chains * self.NumSamples), 1)
        weight_ar3 = weight_ar3.reshape((self.num_chains * self.NumSamples), 1)
        weight_ar4 = weight_ar4.reshape((self.num_chains * self.NumSamples), 1)

        x = np.linspace(0, int(self.masternumsample - self.masternumsample * self.burni),
                        num=int(self.masternumsample - self.masternumsample * self.burni))
        x1 = np.linspace(0, self.masternumsample, num=self.masternumsample)

        plt.plot(x1, weight_ar, label='Weight[0]')
        plt.legend(loc='upper right')
        plt.title("Weight[0] Trace")
        plt.savefig(path + '/weight[0]_samples.png')
        plt.clf()

        plt.hist(weight_ar, bins=20, color="blue", alpha=0.7)
        plt.ylabel('Frequency')
        plt.xlabel('Parameter Values')
        plt.savefig(path + '/weight[0]_hist.png')
        plt.clf()

        plt.plot(x1, weight_ar1, label='Weight[100]')
        plt.legend(loc='upper right')
        plt.title("Weight[100] Trace")
        plt.savefig(path + '/weight[100]_samples.png')
        plt.clf()

        plt.hist(weight_ar1, bins=20, color="blue", alpha=0.7)
        plt.ylabel('Frequency')
        plt.xlabel('Parameter Values')
        plt.savefig(path + '/weight[100]_hist.png')
        plt.clf()

        plt.plot(x1, weight_ar2, label='Weight[50000]')
        plt.legend(loc='upper right')
        plt.title("Weight[50000] Trace")
        plt.savefig(path + '/weight[50000]_samples.png')
        plt.clf()

        plt.hist(weight_ar2, bins=20, color="blue", alpha=0.7)
        plt.ylabel('Frequency')
        plt.xlabel('Parameter Values')
        plt.savefig(path + '/weight[50000]_hist.png')
        plt.clf()

        plt.plot(x1, weight_ar3, label='Weight[40000]')
        plt.legend(loc='upper right')
        plt.title("Weight[40000] Trace")
        plt.savefig(path + '/weight[40000]_samples.png')
        plt.clf()

        plt.hist(weight_ar3, bins=20, color="blue", alpha=0.7)
        plt.ylabel('Frequency')
        plt.xlabel('Parameter Values')
        plt.savefig(path + '/weight[40000]_hist.png')
        plt.clf()

        plt.plot(x1, weight_ar4, label='Weight[60000]')
        plt.legend(loc='upper right')
        plt.title("Weight[60000] Trace")
        plt.savefig(path + '/weight[60000]_samples.png')
        plt.clf()

        plt.hist(weight_ar4, bins=20, color="blue", alpha=0.7)
        plt.ylabel('Frequency')
        plt.xlabel('Parameter Values')
        plt.savefig(path + '/weight[60000]_hist.png')
        plt.clf()

        plt.plot(x1, sum_val_array, label='Sum_Value')
        plt.legend(loc='upper right')
        plt.title("Sum Value Over Samples")
        plt.savefig(path + '/sum_value_samples.png')
        plt.clf()

        color = 'tab:red'
        plt.plot(x1, rmse_train, label="Train", color=color)
        color = 'tab:blue'
        plt.plot(x1, rmse_test, label="Test", color=color)
        plt.xlabel('Samples')
        plt.ylabel('RMSE')
        plt.legend()
        plt.savefig(path + '/superimposed_rmse.png')
        plt.clf()

        return rmse_train, rmse_test, accept_percentage_all_chains

    def make_directory(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

def main():

    numSamples = args.samples
    num_chains = args.num_chains
    swap_ratio = args.swap_ratio
    burn_in = args.burn_in
    learning_rate = args.learning_rate
    step_size = args.step_size
    maxtemp = 2
    use_langevin_gradients = True  # False leaves it as Random-walk proposals. Note that Langevin gradients will take a bit more time computationally
    bi = burn_in
    swap_interval = int(swap_ratio * numSamples / num_chains) # int(swap_ratio * (NumSample/num_chains)) #how ofen you swap neighbours. note if swap is more than Num_samples, its off
    # learn_rate = 0.01  # in case langevin gradients are used. Can select other values, we found small value is ok.

    problemfolder = 'Graph_torch/GNN' # change this to your directory for results output - produces large datasets

    name = ""
    filename = ""

    if not os.path.exists(problemfolder + name):
        os.makedirs(problemfolder + name)
    path = (problemfolder + name)

    timer = time.time()

    pt = ParallelTempering(use_langevin_gradients, learning_rate, num_chains, maxtemp, numSamples,
                           swap_interval, path, batch_size, bi, step_size)

    directories = [path + '/predictions/', path + '/graphs/']
    for d in directories:
        pt.make_directory((filename) + d)

    pt.initialize_chains(burn_in)
    # pos_w, fx_train, fx_test, rmse_train, rmse_test, acc_train, acc_test, likelihood_rep, swap_perc, accept_vec, accept = pt.run_chains()
    rmse_train, rmse_test, accept_percent_all, sp = pt.run_chains()

    timer2 = time.time()

    # list_end = accept_vec.shape[1]
    # accept_ratio = accept_vec[:,  list_end-1:list_end]/list_end
    # accept_per = np.mean(accept_ratio) * 100
    # print(accept_per, ' accept_per')

    timetotal = (timer2 - timer) / 60

    """
    # #PLOTS
    acc_tr = np.mean(acc_train [:])
    acctr_std = np.std(acc_train[:])
    acctr_max = np.amax(acc_train[:])
    acc_tes = np.mean(acc_test[:])
    acctest_std = np.std(acc_test[:])
    acctes_max = np.amax(acc_test[:])
    rmse_tr = np.mean(rmse_train[:])
    rmsetr_std = np.std(rmse_train[:])
    rmsetr_max = np.amax(acc_train[:])
    rmse_tes = np.mean(rmse_test[:])
    rmsetest_std = np.std(rmse_test[:])
    rmsetes_max = np.amax(rmse_test[:])
    """

    burnin = burn_in

    #acc_tr = np.mean(acc_train[int(numSamples * burnin):])
    #acctr_std = np.std(acc_train[int(numSamples * burnin):])
    #acctr_max = np.amax(acc_train[int(numSamples * burnin):])

    #acc_tes = np.mean(acc_test[int(numSamples * burnin):])
    #acctest_std = np.std(acc_test[int(numSamples * burnin):])
    #acctes_max = np.amax(acc_test[int(numSamples * burnin):])

    rmse_tr = np.mean(rmse_train[int(numSamples * burnin):])
    rmsetr_std = np.std(rmse_train[int(numSamples * burnin):])
    rmsetr_max = np.amax(rmse_train[int(numSamples * burnin):])

    rmse_tes = np.mean(rmse_test[int(numSamples * burnin):])
    rmsetest_std = np.std(rmse_test[int(numSamples * burnin):])
    rmsetes_max = np.amax(rmse_test[int(numSamples * burnin):])

    accept_percent_mean = np.mean(accept_percent_all)

    # outres = open(path+'/result.txt', "a+")
    # outres_db = open(path_db+'/result.txt', "a+")
    # resultingfile = open(problemfolder+'/master_result_file.txt','a+')
    # resultingfile_db = open( problemfolder_db+'/master_result_file.txt','a+')
    # xv = name+'_'+ str(run_nb)
    print("\n\n\n\n")
    #print("Train Acc (Mean, Max, Std)")
    #print(acc_tr, acctr_max, acctr_std)
    #print("\n")
    #print("Test Acc (Mean, Max, Std)")
    #print(acc_tes, acctes_max, acctest_std)
    #print("\n")
    print("Train MAE (Mean, Max, Std)")
    print(rmse_tr, rmsetr_max, rmsetr_std)
    print("\n")
    print("Test MAE (Mean, Max, Std)")
    print(rmse_tes, rmsetes_max, rmsetest_std)
    print("\n")
    print("Acceptance Percentage Mean")
    print(accept_percent_mean)
    print("\n")
    print("Swap Percentage")
    print(sp)
    print("\n")
    print("Time (Minutes)")
    print(timetotal)


if __name__ == "__main__": main()

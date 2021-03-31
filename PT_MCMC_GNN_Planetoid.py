import os.path as osp
import os
import argparse
import numpy as np
import random
import copy
import multiprocessing
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import time
import matplotlib as mpl

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv

mpl.use('agg')

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, split='public',transform=T.NormalizeFeatures())
graph_data = dataset[0]

num_train = len(graph_data.y[graph_data.train_mask])
num_test= len(graph_data.y[graph_data.test_mask])

# Initialise and parse command-line inputs

parser = argparse.ArgumentParser(description='PT MCMC CNN')
parser.add_argument('-s', '--samples', help='Number of samples', default=80, dest="samples", type=int)
parser.add_argument('-r', '--replicas', help='Number of chains/replicas, best to have one per availble core/cpu',
                    default=8, dest="num_chains", type=int)
parser.add_argument('-lr', '--learning_rate', help='Learning Rate for Model', dest="learning_rate",
                    default=0.01, type=float)
parser.add_argument('-b', '--burn', help='How many samples to discard before determing posteriors', dest="burn_in",
                    default=0.75, type=float)
parser.add_argument('-pt', '--ptsamples', help='Ratio of PT vs straight MCMC samples to run', dest="pt_samples",
                    default=0.60, type=float)
parser.add_argument('-step', '--step_size', help='Step size for proposals (0.02, 0.05, 0.1 etc)', dest="step_size",
                    default=0.005, type=float)
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()


if args.use_gdc:
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
    graph_data = gdc(graph_data)

class Net(torch.nn.Module):
    def __init__(self, lrate):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True,
                             normalize=not args.use_gdc)
        #self.optimizer = torch.optim.Adam(self.parameters(), lr=lrate, weight_decay=0.0005)
        self.optimizer = torch.optim.Adam([
            dict(params=self.conv1.parameters(),weight_decay=5e-4),
            dict(params=self.conv2.parameters(),weight_decay=0)
        ], lr=lrate)
        #self.optimizer = torch.optim.SGD(self.parameters(), lr=lrate)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self):
        x, edge_index, edge_weight = graph_data.x, graph_data.edge_index, graph_data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

    def evaluate_proposal(self, data, w=None):
        self.los =0
        if w is not None:
            self.loadparameters(w)
        if (data=='train'):
            prob = copy.deepcopy(self.forward().detach())
            for _, mask in graph_data('train_mask'):
                y_pred = prob[mask].max(1)[1]
            loss = F.nll_loss(self.forward()[graph_data.train_mask], graph_data.y[graph_data.train_mask])
            self.los += loss
            prob = prob[mask]
        else:
            prob = copy.deepcopy(self.forward().detach())
            for _, mask in graph_data('test_mask'):
                y_pred = prob[mask].max(1)[1]
            loss = F.nll_loss(self.forward()[graph_data.test_mask], graph_data.y[graph_data.test_mask])
            self.los += loss
            prob = prob[mask]
        return y_pred, prob

    def langevin_gradient(self, w=None):
        if w is not None:
            self.loadparameters(w)
        self.los = 0
        self.optimizer.zero_grad()
        loss = F.nll_loss(self.forward()[graph_data.train_mask], graph_data.y[graph_data.train_mask])
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

class ptReplica(multiprocessing.Process):
    def __init__(self, use_langevin_gradients, learn_rate, w, minlim_param, maxlim_param, samples,
                 burn_in, temperature, swap_interval, path, parameter_queue, main_process, event, step_size):
        self.gnn = Net(learn_rate)
        multiprocessing.Process.__init__(self)
        self.processID = temperature
        self.parameter_queue = parameter_queue
        self.signal_main = main_process
        self.event = event
        self.temperature = temperature
        self.adapttemp = temperature
        self.swap_interval = swap_interval
        self.path = path
        self.burn_in = burn_in
        self.samples = samples
        self.traindata = 'train'
        self.testdata = 'test'
        self.w = w
        self.minY = np.zeros((1, 1))
        self.maxY = np.zeros((1, 1))
        self.minlim_param = minlim_param
        self.maxlim_param = maxlim_param
        self.use_langevin_gradients = use_langevin_gradients
        self.sgd_depth = 1  # Keep as 1
        self.learn_rate = learn_rate
        self.l_prob = 1.0  # Ratio of langevin based proposals, higher value leads to more computation time, evaluate for different problems
        self.step_size = step_size

    def rmse(self, predictions, targets):
        return self.gnn.los.item()

    @staticmethod
    def likelihood_func(gnn, data, temp, w=None):
        if w is not None:
            fx, prob = gnn.evaluate_proposal(data, w)
        else:
            fx, prob = gnn.evaluate_proposal(data)

        if (data == 'train'):
            y = graph_data.y[graph_data.train_mask]
            rmse = gnn.los / num_train
            lhood = 0
            for i in range(num_train):
                for k in range(dataset.num_classes):
                    if k == y[i]:
                        if prob[i, k] == 0:
                            lhood+=0
                        else:
                            lhood += (prob[i,k])
        else:
            y = graph_data.y[graph_data.test_mask]
            rmse = gnn.los / num_test
            lhood = 0
            for i in range(num_test):
                for k in range(dataset.num_classes):
                    if k == y[i]:
                        if prob[i, k] == 0:
                            lhood += 0
                        else:
                            lhood += (prob[i, k])

        return [lhood / temp , fx, rmse]

    def prior_likelihood(self, sigma_squared, w_list):
        part1 = -1 * ((len(w_list)) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w_list)))
        log_loss = part1 - part2
        return log_loss

    def accuracy(self, data):
        gnn =self.gnn
        if (data == 'train'):
            prob = copy.deepcopy(gnn().detach())
            for _, mask in graph_data('train_mask'):
                pred = prob[mask].max(1)[1]
                acc = pred.eq(graph_data.y[mask]).sum().item() / mask.sum().item()
        else:
            prob = copy.deepcopy(gnn().detach())
            for _, mask in graph_data('test_mask'):
                pred = prob[mask].max(1)[1]
                acc = pred.eq(graph_data.y[mask]).sum().item() / mask.sum().item()
        return 100 * acc

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
        likelihood_proposal_array = np.zeros(samples)
        likelihood_array = np.zeros(samples)
        diff_likelihood_array = np.zeros(samples)
        weight_array = np.zeros(samples)
        weight_array1 = np.zeros(samples)
        weight_array2 = np.zeros(samples)
        weight_array3 = np.zeros(samples)
        weight_array4 = np.zeros(samples)
        sum_value_array = np.zeros(samples)

        eta = 0 #junk

        w_proposal = np.random.randn(w_size)
        w_proposal = gnn.dictfromlist(w_proposal)
        train = 'train'
        test = 'test'

        sigma_squared = 25
        prior_current = self.prior_likelihood(sigma_squared, gnn.getparameters(w))

        [likelihood, pred_train, rmsetrain] = self.likelihood_func(gnn, train, self.adapttemp)
        [_, pred_test, rmsetest] = self.likelihood_func(gnn, test,self.adapttemp)

        num_accepted = 0
        langevin_count = 0
        pt_samples = samples * 0.6  # PT in canonical form with adaptive temp will work till assigned limit
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
        weight_array3[0] = 0
        weight_array4[0] = 0

        sum_value_array[0] = 0

        for i in range(
                samples):  # Begin sampling --------------------------------------------------------------------------

            if i < pt_samples:
                self.adapttemp = self.temperature  # T1=T/log(k+1);
            if i == pt_samples and init_count == 0:  # Move to canonical MCMC
                self.adapttemp = 1
                [likelihood, pred_train, rmsetrain] = self.likelihood_func(gnn, train, self.adapttemp, w)
                [_, pred_test, rmsetest] = self.likelihood_func(gnn, test, self.adapttemp, w)
                init_count = 1

            lx = np.random.uniform(0, 1, 1)
            old_w = gnn.state_dict()

            if (self.use_langevin_gradients is True) and (lx < self.l_prob):
                w_gd = gnn.langevin_gradient()  # Eq 8
                w_proposal = gnn.addnoiseandcopy(0, step_w)  # np.random.normal(w_gd, step_w, w_size) # Eq 7
                w_prop_gd = gnn.langevin_gradient()
                wc_delta = (gnn.getparameters(w) - gnn.getparameters(w_prop_gd))
                wp_delta = (gnn.getparameters(w_proposal) - gnn.getparameters(w_gd))
                sigma_sq = step_w * step_w
                first = -0.5 * np.sum(wc_delta * wc_delta) / sigma_sq  # this is wc_delta.T  *  wc_delta /sigma_sq
                second = -0.5 * np.sum(wp_delta * wp_delta) / sigma_sq
                diff_prop = first - second
                diff_prop = diff_prop / self.adapttemp
                langevin_count = langevin_count + 1
            else:
                diff_prop = 0
                w_proposal = gnn.addnoiseandcopy(0, step_w)  # np.random.normal(w, step_w, w_size)

            [likelihood_proposal, pred_train, rmsetrain] = self.likelihood_func(gnn, train,self.adapttemp)
            [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(gnn, test,self.adapttemp)

            prior_prop = self.prior_likelihood(sigma_squared,
                                               gnn.getparameters(w_proposal))  # takes care of the gradients
            diff_likelihood = likelihood_proposal - likelihood
            diff_prior = prior_prop - prior_current

            likelihood_proposal_array[i] = likelihood_proposal
            likelihood_array[i] = likelihood
            diff_likelihood_array[i] = diff_likelihood

            sum_value = diff_likelihood + diff_prior + diff_prop
            sum_value_array[i] = sum_value
            u = np.log(random.uniform(0, 1))

            if u < sum_value:
                num_accepted = num_accepted + 1
                likelihood = likelihood_proposal
                prior_current = prior_prop
                w = copy.deepcopy(w_proposal)  # rnn.getparameters(w_proposal)
                acc_train1 = self.accuracy(train)
                acc_test1 = self.accuracy(test)
                print (i, rmsetrain, rmsetest, acc_train1, acc_test1, 'accepted')
                rmse_train[i] = rmsetrain
                rmse_test[i] = rmsetest
                acc_train[i,] = acc_train1
                acc_test[i,] = acc_test1

            else:
                w = old_w
                gnn.loadparameters(w)
                acc_train1 = self.accuracy(train)
                acc_test1 = self.accuracy(test)
                print (i, rmsetrain, rmsetest, acc_train1, acc_test1, 'rejected')
                rmse_train[i,] = rmse_train[i - 1,]
                rmse_test[i,] = rmse_test[i - 1,]
                acc_train[i,] = acc_train[i - 1,]
                acc_test[i,] = acc_test[i - 1,]

            ll = gnn.getparameters()
            #print(ll.size)
            weight_array[i] = ll[0]
            weight_array1[i] = ll[100]
            weight_array2[i] = ll[1000]
            weight_array3[i] = ll[5000]
            weight_array4[i] = ll[8000]

            if (i + 1) % self.swap_interval == 0:
                param = np.concatenate([np.asarray([gnn.getparameters(w)]).reshape(-1), np.asarray([eta]).reshape(-1),
                                        np.asarray([likelihood]), np.asarray([self.adapttemp]), np.asarray([i])])
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
             np.asarray([self.adapttemp]), np.asarray([i])])

        self.signal_main.set()

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

        file_name = self.path + '/predictions/weight[1000]_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, weight_array2, fmt='%1.2f')

        file_name = self.path + '/predictions/weight[5000]_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, weight_array3, fmt='%1.2f')

        file_name = self.path + '/predictions/weight[8000]_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, weight_array4, fmt='%1.2f')

        file_name = self.path + '/predictions/rmse_test_chain_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, rmse_test, fmt='%1.2f')

        file_name = self.path + '/predictions/rmse_train_chain_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, rmse_train, fmt='%1.2f')

        file_name = self.path + '/predictions/acc_test_chain_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, acc_test, fmt='%1.2f')

        file_name = self.path + '/predictions/acc_train_chain_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, acc_train, fmt='%1.2f')

        file_name = self.path + '/predictions/accept_percentage' + str(self.temperature) + '.txt'
        with open(file_name, 'w') as f:
            f.write('%d' % accept_ratio)

        file_name = self.path + '/likelihood_value_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, likelihood_array, fmt='%1.4f')

# Manages the parallel tempering, initialises and executes the parallel chains

class ParallelTempering:
    def __init__(self, use_langevin_gradients, learn_rate, num_chains, maxtemp, NumSample, swap_interval,
                 path, bi, step_size):
        gnn = Net(learn_rate)
        self.gnn = gnn
        self.traindata = 'train'
        self.testdata = 'test'
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
                          self.NumSamples, self.burn_in, self.temperatures[i], self.swap_interval, self.path,
                          self.parameter_queue[i], self.wait_chain[i], self.event[i], self.step_size))

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
        w1 = self.gnn.dictfromlist(w1)
        T1 = param1[self.num_param + 2]
        lhood1 = param1[self.num_param + 1]
        w2 = param2[0:self.num_param]
        w2 = self.gnn.dictfromlist(w2)
        lhood2 = param2[self.num_param + 1]
        T2 = param2[self.num_param + 2]
        # SWAPPING PROBABILITIES
        lhood12, dump1, dump2 = ptReplica.likelihood_func(self.gnn, self.traindata, T2, w1)
        lhood21, dump1, dump2 = ptReplica.likelihood_func(self.gnn, self.traindata, T1, w2)
        try:
            swap_proposal = min(1, np.exp((lhood12 - lhood1) + (lhood21 - lhood2)))
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
            param1[self.num_param + 1] = lhood21
            param2[self.num_param + 1] = lhood12
            param1[self.num_param + 2] = T2
            param2[self.num_param + 2] = T1
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
        #start_time = time.time()
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
            #print("--- %s seconds ---" % (time.time() - start_time))

        print("Joining Processes")

        # JOIN THEM TO MAIN PROCESS
        for index in range(0, self.num_chains):
            print('Waiting to Join ', index, self.num_chains)
            print(self.chains[index].is_alive())
            self.chains[index].join()
            print(index, 'Chain Joined')
        self.chain_queue.join()
        # pos_w, fx_train, fx_test, rmse_train, rmse_test, acc_train, acc_test, likelihood_vec, accept_vec, accept = self.show_results()
        rmse_train, rmse_test, acc_train, acc_test, apal = self.show_results()
        print("NUMBER OF SWAPS = ", self.num_swap)
        swap_perc = self.num_swap * 100 / self.total_swap_proposals
        # return pos_w, fx_train, fx_test, rmse_train, rmse_test, acc_train, acc_test, likelihood_vec, swap_perc, accept_vec, accept
        return rmse_train, rmse_test, acc_train, acc_test, apal, swap_perc

    def show_results(self):
        burnin = int(self.NumSamples * self.burn_in)
        mcmc_samples = int(self.NumSamples * 0.25)
        # likelihood_rep = np.zeros((self.num_chains, self.NumSamples - burnin,2))  # index 1 for likelihood posterior and index 0 for Likelihood proposals. Note all likilihood proposals plotted only
        # accept_percent = np.zeros((self.num_chains, 1))
        # accept_list = np.zeros((self.num_chains, self.NumSamples))
        # pos_w = np.zeros((self.num_chains, self.NumSamples - burnin, self.num_param))
        # fx_train_all = np.zeros((self.num_chains, self.NumSamples - burnin, len(self.traindata)))
        rmse_train = np.zeros((self.num_chains, self.NumSamples))
        acc_train = np.zeros((self.num_chains, self.NumSamples))

        # fx_test_all = np.zeros((self.num_chains, self.NumSamples - burnin, len(self.testdata)))
        rmse_test = np.zeros((self.num_chains, self.NumSamples))
        acc_test = np.zeros((self.num_chains, self.NumSamples))
        sum_val_array = np.zeros((self.num_chains, self.NumSamples))

        weight_ar = np.zeros((self.num_chains, self.NumSamples))
        weight_ar1 = np.zeros((self.num_chains, self.NumSamples))
        weight_ar2 = np.zeros((self.num_chains, self.NumSamples))
        weight_ar3 = np.zeros((self.num_chains, self.NumSamples))
        weight_ar4 = np.zeros((self.num_chains, self.NumSamples))
        likelihood_val_array = np.zeros((self.num_chains, self.NumSamples))

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

            file_name = self.path + '/predictions/acc_test_chain_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            acc_test[i, :] = dat

            file_name = self.path + '/predictions/acc_train_chain_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            acc_train[i, :] = dat

            file_name = self.path + '/predictions/sum_value_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            sum_val_array[i, :] = dat

            file_name = self.path + '/predictions/weight[0]_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            weight_ar[i, :] = dat

            file_name = self.path + '/predictions/weight[100]_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            weight_ar1[i, :] = dat

            file_name = self.path + '/predictions/weight[1000]_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            weight_ar2[i, :] = dat

            file_name = self.path + '/predictions/weight[5000]_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            weight_ar3[i, :] = dat

            file_name = self.path + '/predictions/weight[8000]_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            weight_ar4[i, :] = dat

            file_name = self.path + '/predictions/accept_percentage' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            accept_percentage_all_chains[i] = dat

            file_name = self.path + '/likelihood_value_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            likelihood_val_array[i, :] = dat

        rmse_train_single_chain_plot = rmse_train[0, :]
        rmse_test_single_chain_plot = rmse_test[0, :]
        acc_train_single_chain_plot = acc_train[0, :]
        acc_test_single_chain_plot = acc_test[0, :]
        sum_val_array_single_chain_plot = sum_val_array[0]
        likelihood_val_array_single_chain_plot = likelihood_val_array[0]

        #path = 'GNN/graphs'

        x2 = np.linspace(0, self.NumSamples, num=self.NumSamples)

        plt.plot(x2, sum_val_array_single_chain_plot, label='Sum Value')
        plt.legend(loc='upper right')
        plt.savefig(self.path + '/graphs/sum_value_single_chain.png')
        plt.clf()

        plt.plot(x2, likelihood_val_array_single_chain_plot, label='Sum Value')
        plt.legend(loc='upper right')
        plt.ylabel("Likelihood", fontsize=13)
        plt.xlabel("Samples", fontsize=13)
        plt.yticks(fontsize=13)
        plt.xticks(fontsize=13)
        plt.savefig(self.path + '/likelihood_value_single_chain.png')
        plt.clf()

        color = 'tab:red'
        plt.plot(x2, acc_train_single_chain_plot, label="Train", color=color)
        color = 'tab:blue'
        plt.plot(x2, acc_test_single_chain_plot, label="Test", color=color)
        plt.xlabel('Samples',fontsize=13)
        plt.ylabel('Accuracy',fontsize=13)
        plt.yticks(fontsize=13)
        plt.xticks(fontsize=13)
        plt.legend()
        plt.savefig(self.path + '/graphs/superimposed_acc_single_chain.png')
        plt.clf()

        color = 'tab:red'
        plt.plot(x2, rmse_train_single_chain_plot, label="Train", color=color)
        color = 'tab:blue'
        plt.plot(x2, rmse_test_single_chain_plot, label="Test", color=color)
        plt.xlabel('Samples')
        plt.ylabel('RMSE')
        plt.legend()
        plt.savefig(self.path + '/graphs/superimposed_rmse_single_chain.png')
        plt.clf()

        rmse_train = rmse_train.reshape((self.num_chains * self.NumSamples), 1)
        acc_train = acc_train.reshape((self.num_chains * self.NumSamples), 1)
        rmse_test = rmse_test.reshape((self.num_chains * self.NumSamples), 1)
        acc_test = acc_test.reshape((self.num_chains * self.NumSamples), 1)
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
        plt.ylabel('Parameter Values', fontsize=13)
        plt.xlabel('Samples', fontsize=13)
        plt.yticks(fontsize=13)
        plt.xticks(fontsize=13)
        plt.savefig(self.path + '/graphs/weight[0]_samples.png')
        plt.clf()

        plt.hist(weight_ar, bins=20, color="blue", alpha=0.7)
        plt.ylabel('Frequency', fontsize=13)
        plt.xlabel('Parameter Values', fontsize=13)
        plt.yticks(fontsize=13)
        plt.xticks(fontsize=13)
        plt.savefig(self.path + '/graphs/weight[0]_hist.png')
        plt.clf()

        plt.plot(x1, weight_ar1, label='Weight[100]')
        plt.legend(loc='upper right')
        plt.ylabel('Parameter Values', fontsize=13)
        plt.xlabel('Samples', fontsize=13)
        plt.yticks(fontsize=13)
        plt.xticks(fontsize=13)
        plt.savefig(self.path + '/graphs/weight[100]_samples.png')
        plt.clf()

        plt.hist(weight_ar1, bins=20, color="blue", alpha=0.7)
        plt.ylabel('Frequency', fontsize=13)
        plt.xlabel('Parameter Values', fontsize=13)
        plt.yticks(fontsize=13)
        plt.xticks(fontsize=13)
        plt.savefig(self.path + '/graphs/weight[100]_hist.png')
        plt.clf()

        plt.plot(x1, weight_ar2, label='Weight[1000]')
        plt.legend(loc='upper right')
        plt.ylabel('Parameter Values', fontsize=13)
        plt.xlabel('Samples', fontsize=13)
        plt.yticks(fontsize=13)
        plt.xticks(fontsize=13)
        plt.savefig(self.path + '/graphs/weight[1000]_samples.png')
        plt.clf()

        plt.hist(weight_ar2, bins=20, color="blue", alpha=0.7)
        plt.ylabel('Frequency', fontsize=13)
        plt.xlabel('Parameter Values', fontsize=13)
        plt.yticks(fontsize=13)
        plt.xticks(fontsize=13)
        plt.savefig(self.path + '/graphs/weight[1000]_hist.png')
        plt.clf()

        plt.plot(x1, weight_ar3, label='Weight[5000]')
        plt.legend(loc='upper right')
        plt.ylabel('Parameter Values', fontsize=13)
        plt.xlabel('Samples', fontsize=13)
        plt.yticks(fontsize=13)
        plt.xticks(fontsize=13)
        plt.savefig(self.path + '/graphs/weight[5000]_samples.png')
        plt.clf()

        plt.hist(weight_ar3, bins=20, color="blue", alpha=0.7)
        plt.ylabel('Frequency', fontsize=13)
        plt.xlabel('Parameter Values', fontsize=13)
        plt.yticks(fontsize=13)
        plt.xticks(fontsize=13)
        plt.savefig(self.path + '/graphs/weight[5000]_hist.png')
        plt.clf()

        plt.plot(x1, weight_ar4, label='Weight[8000]')
        plt.legend(loc='upper right')
        plt.ylabel('Parameter Values', fontsize=13)
        plt.xlabel('Samples', fontsize=13)
        plt.yticks(fontsize=13)
        plt.xticks(fontsize=13)
        plt.savefig(self.path + '/graphs/weight[8000]_samples.png')
        plt.clf()

        plt.hist(weight_ar4, bins=20, color="blue", alpha=0.7)
        plt.ylabel('Frequency', fontsize=13)
        plt.xlabel('Parameter Values', fontsize=13)
        plt.yticks(fontsize=13)
        plt.xticks(fontsize=13)
        plt.savefig(self.path + '/graphs/weight[8000]_hist.png')
        plt.clf()

        plt.plot(x1, sum_val_array, label='Sum_Value')
        plt.legend(loc='upper right')
        plt.title("Sum Value Over Samples")
        plt.savefig(self.path + '/graphs/sum_value_samples.png')
        plt.clf()

        color = 'tab:red'
        plt.plot(x1, acc_train, label="Train", color=color)
        color = 'tab:blue'
        plt.plot(x1, acc_test, label="Test", color=color)
        plt.xlabel('Samples')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(self.path + '/graphs/superimposed_acc.png')
        plt.clf()

        color = 'tab:red'
        plt.plot(x1, rmse_train, label="Train", color=color)
        color = 'tab:blue'
        plt.plot(x1, rmse_test, label="Test", color=color)
        plt.xlabel('Samples')
        plt.ylabel('RMSE')
        plt.legend()
        plt.savefig(self.path + '/graphs/superimposed_rmse.png')
        plt.clf()

        return rmse_train, rmse_test, acc_train, acc_test, accept_percentage_all_chains

    def make_directory(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

def main():

    numSamples = args.samples
    num_chains = args.num_chains
    burn_in = args.burn_in
    learning_rate = args.learning_rate
    step_size = args.step_size
    maxtemp = 2
    use_langevin_gradients = True  # False leaves it as Random-walk proposals. Note that Langevin gradients will take a bit more time computationally
    bi = burn_in
    swap_interval = 2 #how ofen you swap neighbours. note if swap is more than Num_samples, its off

    # learn_rate = 0.01  # in case langevin gradients are used. Can select other values, we found small value is ok.

    problemfolder = 'Graph_torch/GNN'  # change this to your directory for results output - produces large datasets

    name = ""
    filename = ""

    if not os.path.exists(problemfolder + name):
        os.makedirs(problemfolder + name)
    path = (problemfolder + name)

    timer = time.time()

    pt = ParallelTempering(use_langevin_gradients, learning_rate, num_chains, maxtemp, numSamples,
                           swap_interval, path, bi, step_size)

    directories = [path + '/predictions/', path + '/graphs/']
    for d in directories:
        pt.make_directory((filename) + d)

    pt.initialize_chains(burn_in)
    # pos_w, fx_train, fx_test, rmse_train, rmse_test, acc_train, acc_test, likelihood_rep, swap_perc, accept_vec, accept = pt.run_chains()
    rmse_train, rmse_test, acc_train, acc_test, accept_percent_all, sp = pt.run_chains()

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

    acc_tr = np.mean(acc_train[int(numSamples * burnin):])
    acctr_std = np.std(acc_train[int(numSamples * burnin):])
    acctr_max = np.amax(acc_train[int(numSamples * burnin):])

    acc_tes = np.mean(acc_test[int(numSamples * burnin):])
    acctest_std = np.std(acc_test[int(numSamples * burnin):])
    acctes_max = np.amax(acc_test[int(numSamples * burnin):])

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
    print("Train Acc (Mean, Max, Std)")
    print(acc_tr, acctr_max, acctr_std)
    print("\n")
    print("Test Acc (Mean, Max, Std)")
    print(acc_tes, acctes_max, acctest_std)
    print("\n")
    print("Train RMSE (Mean, Max, Std)")
    print(rmse_tr, rmsetr_max, rmsetr_std)
    print("\n")
    print("Test RMSE (Mean, Max, Std)")
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

import os.path as osp
import os
import argparse
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, split='public', num_train_per_class=242,num_val=0,num_test=1014, transform=T.NormalizeFeatures())
graph_data = dataset[0]

num_class = 7
num_train = len(graph_data.y[graph_data.train_mask])
num_test= len(graph_data.y[graph_data.test_mask])
print(num_train)
print(num_test)

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
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lrate)
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

class MCMC:
    def __init__(self, samples, use_langevin_gradients, lr):
        self.samples = samples
        self.gnn = Net(lr)
        self.traindata = 'train'
        self.testdata = 'test'
        self.use_langevin_gradients = use_langevin_gradients
        self.l_prob=0.7

    def rmse(self, predictions, targets):
        return self.gnn.los.item()

    def likelihood_func(self, gnn, data, w=None):
        if w is not None:
            fx, prob = gnn.evaluate_proposal(data, w)
        else:
            fx, prob = gnn.evaluate_proposal(data)

        if (data == 'train'):
            y = graph_data.y[graph_data.train_mask]
            rmse = self.gnn.los / num_train
            lhood = 0
            for i in range(num_train):
                for k in range(num_class):
                    if k == y[i]:
                        if prob[i, k] == 0:
                            lhood+=0
                        else:
                            lhood += (prob[i,k])
        else:
            y = graph_data.y[graph_data.test_mask]
            rmse = self.gnn.los / num_test
            lhood = 0
            for i in range(num_test):
                for k in range(num_class):
                    if k == y[i]:
                        if prob[i, k] == 0:
                            lhood += 0
                        else:
                            lhood += (prob[i, k])

        return [lhood, fx, rmse]

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
        weight_array3 = np.zeros(samples)
        sum_value_array=np.zeros(samples)

        w_proposal = np.random.randn(w_size)
        w_proposal = gnn.dictfromlist(w_proposal)
        step_w = 0.005
        train = 'train'
        test = 'test'
        sigma_squared = 25
        prior_current = self.prior_likelihood(sigma_squared, gnn.getparameters(w))

        [likelihood, pred_train, rmsetrain] = self.likelihood_func(gnn, train)
        [_, pred_test, rmsetest] = self.likelihood_func(gnn, test)

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
        weight_array3[0] = 0
        sum_value_array[0] = 0

        for i in range(samples):  # Begin sampling --------------------------------------------------------------------------

            lx = np.random.uniform(0, 1, 1)
            old_w = gnn.state_dict()

            if (self.use_langevin_gradients is True) and (lx < self.l_prob):
                w_gd = gnn.langevin_gradient()  # Eq 8
                w_proposal = gnn.addnoiseandcopy(0, step_w)  # np.random.normal(w_gd, step_w, w_size) # Eq 7
                w_prop_gd = gnn.langevin_gradient()
                wc_delta = (gnn.getparameters(w) - gnn.getparameters(w_prop_gd))
                wp_delta = (gnn.getparameters(w_proposal) - gnn.getparameters(w_gd))
                sigma_sq = step_w
                first = -0.5 * np.sum(wc_delta * wc_delta) / sigma_sq  # this is wc_delta.T  *  wc_delta /sigma_sq
                second = -0.5 * np.sum(wp_delta * wp_delta) / sigma_sq
                diff_prop = first - second
                langevin_count = langevin_count + 1
            else:
                diff_prop = 0
                w_proposal = gnn.addnoiseandcopy(0, step_w)  # np.random.normal(w, step_w, w_size)

            [likelihood_proposal, pred_train, rmsetrain] = self.likelihood_func(gnn, train)
            [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(gnn, test)

            prior_prop = self.prior_likelihood(sigma_squared, gnn.getparameters(w_proposal))  # takes care of the gradients

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

            ll=gnn.getparameters()
            weight_array[i]=ll[0]
            weight_array1[i] = ll[1]
            weight_array2[i] = ll[2]
            weight_array3[i] = ll[3]

        file_name = 'weight[0]' + '.txt'
        np.savetxt(file_name, weight_array, fmt='%1.6f')

        file_name = 'weight[1]' + '.txt'
        np.savetxt(file_name, weight_array1, fmt='%1.6f')

        file_name = 'weight[2]' + '.txt'
        np.savetxt(file_name, weight_array2, fmt='%1.6f')

        file_name = 'weight[3]' + '.txt'
        np.savetxt(file_name, weight_array3, fmt='%1.6f')

        print((num_accepted * 100 / (samples * 1.0)), '% was Accepted')

        print((langevin_count * 100 / (samples * 1.0)), '% was Langevin')

        return acc_train, acc_test, rmse_train, rmse_test, sum_value_array, weight_array, weight_array1, weight_array2, weight_array3

def main():


    numSamples = 50

    ulg = True

    lrate=0.001
    burnin =0.25

    mcmc = MCMC(numSamples, ulg, lrate)  # declare class
    acc_train, acc_test, rmse_train, rmse_test, sva, wa, wa1, wa2, wa3 = mcmc.sampler()

    acc_train=acc_train[int(numSamples*burnin):]
    acc_test=acc_test[int(numSamples*burnin):]
    rmse_train=rmse_train[int(numSamples*burnin):]
    rmse_test=rmse_test[int(numSamples*burnin):]
    sva=sva[int(numSamples*burnin):]

    print("\n\n\n\n\n\n\n\n")
    print("Mean of RMSE Train")
    print(np.mean(rmse_train))
    print("\n")
    print("Mean of Accuracy Train")
    print(np.mean(acc_train))
    print("\n")
    print("Mean of RMSE Test")
    print(np.mean(rmse_test))
    print("\n")
    print("Mean of Accuracy Test")
    print(np.mean(acc_test))
    print ('sucessfully sampled')

    problemfolder = 'Cora_single_chain'
    os.makedirs(problemfolder)

    x = np.linspace(0, int(numSamples-numSamples*burnin), num=int(numSamples-numSamples*burnin)+1)
    x1 = np.linspace(0, numSamples, num=numSamples)

    plt.plot(x1, wa, label='Weight[0]')
    plt.legend(loc='upper right')
    plt.title("Weight[0] Trace")
    plt.savefig('Cora_single_chain' + '/weight[0]_samples.png')
    plt.clf()

    plt.plot(x1, wa1, label='Weight[1]')
    plt.legend(loc='upper right')
    plt.title("Weight[100] Trace")
    plt.savefig('Cora_single_chain' + '/weight[1]_samples.png')
    plt.clf()

    plt.plot(x1,wa2, label='Weight[2]')
    plt.legend(loc='upper right')
    plt.title("Weight[50000] Trace")
    plt.savefig('Cora_single_chain' + '/weight[2]_samples.png')
    plt.clf()

    plt.plot(x1, wa3, label='Weight[3]')
    plt.legend(loc='upper right')
    plt.title("Weight[10000] Trace")
    plt.savefig('Cora_single_chain' + '/weight[3]_samples.png')
    plt.clf()

    plt.plot(x, sva, label='Sum_Value')
    plt.legend(loc='upper right')
    plt.title("Sum Value Over Samples")
    plt.savefig('Cora_single_chain'+'/sum_value_samples.png')
    plt.clf()

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Samples')
    ax1.set_ylabel('Accuracy Train', color=color)
    ax1.plot(x, acc_train, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Accuracy Test', color=color)  # we already handled the x-label with ax1
    ax2.plot(x, acc_test, color=color)
    ax2.tick_params(axis='y', labelcolor=color)


    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('Cora_single_chain' + '/superimposed_acc.png')
    plt.clf()

    fig1, ax4 = plt.subplots()

    color = 'tab:red'
    ax4.set_xlabel('Samples')
    ax4.set_ylabel('RMSE Train', color=color)
    ax4.plot(x, rmse_train, color=color)
    ax4.tick_params(axis='y', labelcolor=color)

    ax5 = ax4.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax5.set_ylabel('RMSE Test', color=color)  # we already handled the x-label with ax1
    ax5.plot(x, rmse_test, color=color)
    ax5.tick_params(axis='y', labelcolor=color)


    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('Cora_single_chain' + '/superimposed_rmse.png')
    plt.clf()



if __name__ == "__main__": main()

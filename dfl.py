import numpy as np
import copy
import argparse
from torch.utils.data import DataLoader, Dataset
from torch.autograd import grad
from torch import nn, tensor, mul
from torch import pow as torchpow
from torch import sum as torchsum
from torchvision import datasets, transforms
import networkx as nx
import matplotlib.pyplot as plt
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
import random

from models import MLP, CNNMnist, Linear, Line
from data import custom_datasets

class DataHandler(object):
    def __init__(self, dataset:str, num_users:int, iid:bool):
        # load dataset and split users
        self.n = num_users
        if dataset == 'mnist':
            self.train, self.test, self.idxs = self.mnist(iid, num_users)
            self.output_dim = 10
        elif dataset == "epsilon":
            self.train, self.test, self.idxs = self.epsilon(iid, num_users)
            self.output_dim = 2
        elif dataset == "line":
            self.train, self.test, self.idxs = self.line(iid, num_users)
            self.output_dim = 1
        elif dataset == "2clusters":
            self.output_dim = 2
            self.train, self.test, self.idxs = self.clusters(iid, num_users, num_clusters=self.output_dim)
        elif dataset == "4clusters":
            self.output_dim = 4
            self.train, self.test, self.idxs = self.clusters(iid, num_users, num_clusters=self.output_dim)
        else:
            exit('Error: unrecognized dataset')
        self.data_dim = self.train[0][0].shape
    
    def clusters(self, IID:bool, n:int, num_clusters=2, samples_per_user_train=100, samples_per_user_test=100):
        if IID:
            train = custom_datasets.Clusters(size=n*samples_per_user_train, num_clusters=num_clusters)
            idxs = self.iid(train, n)
        else:
            if n==1:
                exit("Don't use IID=False if only one device!")
            train = custom_datasets.Clusters(size=n*samples_per_user_train, num_clusters=num_clusters)  
            idxs = self.noniid(train.targets.numpy().astype(int).flatten(), n, shards_per_user=1)
        test = custom_datasets.Clusters(size=n*samples_per_user_test, num_clusters=num_clusters, cluster_centers=train.cluster_centers)
        return train, test, idxs
    
    def line(self, IID:bool, n:int, samples_per_user_train=100, samples_per_user_test=100):
        if IID:
            biases = [0]
            train = custom_datasets.Line(size=n*samples_per_user_train, biases=biases)
            idxs = self.iid(train, n)
        else:
            if n==1:
                exit("Don't use IID=False if only one device!")
            biases = list(np.random.uniform(low=-1, high=5, size=(n)))
            train = custom_datasets.Line(size=n*samples_per_user_train, biases=biases)
            shards = [i for i in range(n)]
            idxs = {}
            for i in range(n):
                start = np.random.choice(shards, 1, replace=False)
                shards = list(set(shards) - set(start))
                idxs[i] = [int(start*samples_per_user_train + k) for k in range(samples_per_user_train)]
        test = custom_datasets.Line(size=n*samples_per_user_test, biases=biases)
        return train, test, idxs

    def epsilon(self, IID:bool, n:int):
        if IID:
            train = custom_datasets.EpsilonNormalized(train=True, sort=False)
            idxs = self.iid(train, n)
        else:
            train = custom_datasets.EpsilonNormalized(train=True, sort=True)            
            idxs = self.noniid(train.targets.numpy().astype(int), n)
        test = custom_datasets.EpsilonNormalized(train=False, sort=False)
        return train, test, idxs

    def mnist(self, IID:bool, n:int):
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        if IID:
            idxs = self.iid(train, n)
        else:
            idxs = self.noniid(train.targets.numpy(), n)
        return train, test, idxs

    def iid(self, dataset, n:int):
        """
        Sample I.I.D. client data from dataset
        """
        num_items = int(len(dataset)/n)
        idxs, all_idxs = {}, [i for i in range(len(dataset))]
        for i in range(n):
            idxs[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - idxs[i])
        return idxs

    def noniid(self, labels:list, n:int, shards_per_user=2):
        """
        Sample non-I.I.D client data from MNIST dataset
        """
        num_shards = n*shards_per_user
        samples_per_shard = int(len(labels)/num_shards)
        assert(samples_per_shard*num_shards==len(labels), "Split into shards must give same size as number of labels")
        idx_shard = [i for i in range(num_shards)]
        idxs = {i: np.array([], dtype='int64') for i in range(n)}
        sample_idxs = np.arange(num_shards*samples_per_shard)

        # sort labels
        idxs_labels = np.vstack((sample_idxs, labels))
        idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
        sample_idxs = idxs_labels[0,:]

        # divide and assign
        for i in range(n):
            rand_set = set(np.random.choice(idx_shard, shards_per_user, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                idxs[i] = np.concatenate((idxs[i], sample_idxs[rand*samples_per_shard:(rand+1)*samples_per_shard]), axis=0)
        return idxs

class ModelHandler(object):
    def __init__(self, args, data):
        self.local_models = []
        input_size = data.data_dim
        output_size = data.output_dim
        n = args.num_users
        for i in range(n):
            if args.model == 'cnn' and args.dataset == 'mnist':
                local_model = CNNMnist(args=args).to(args.device)
            elif args.model == 'mlp' and (args.dataset in ['mnist']):
                len_in = 1
                for x in input_size:
                    len_in *= x
                local_model = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
            elif args.model == "line" and (args.dataset in ['line']):
                local_model = Line().to(args.device)
            elif args.model == "linear_model" and (args.dataset in ['line']):
                len_in = 1
                for x in input_size:
                    len_in *= x
                local_model = Linear(i=len_in).to(args.device)
            elif args.model == "linear_model" and (args.dataset in ["2clusters", "4clusters", 'epsilon']):
                len_in = 1
                for x in input_size:
                    len_in *= x
                local_model = Linear(i=len_in, o=output_size)
            else:
                exit('Error: unrecognized model')
            self.local_models.append(local_model)

    def get_local_models(self):
        return self.local_models

class GraphHandler(object):
    def __init__(self, n = 1, topology="random", A = None, r=2, h=1, p = 0):
        self.n = n

        if topology == "custom":
            self.G = nx.from_numpy_array(A) 
        elif topology == "erdos_random":
            # A good guideline around p = logn/n for connectedness 
            if not p:
                p = np.log(n)/n
                print("P was set to 0, setting to p=logn/n={}".format(p))
            er = nx.erdos_renyi_graph(n, p)
            self.G = er.subgraph(max(nx.connected_components(er), key=len))
        elif topology == "geometric_random":
            if not p:
                p = 0.37
                print("R was set to 0, setting to r={}".format(p))
            gr = nx.random_geometric_graph(n, p)
            self.G = gr.subgraph(max(nx.connected_components(gr), key=len))
        elif topology == "complete":
            self.G = nx.complete_graph(self.n) 
        elif topology =="tree":
            self.G = nx.balanced_tree(r, h)
        elif topology == "ring":
            graph = nx.Graph()
            sources = np.arange(self.n)
            for i in range(self.n):
                graph.add_node(i)
            for i in range(1, 2):
                targets = np.roll(sources, i)
                graph.add_edges_from(zip(sources, targets))
            self.G = graph
        elif topology == "eye":
            self.G = nx.from_numpy_array(np.zeros((self.n,self.n)))
        elif topology == "star":
            self.G = nx.star_graph(self.n-1)
        else:
            exit("{} is an unknown topology!".format(topology))

        self.A = nx.to_numpy_array(self.G)
        if np.sum(np.multiply(self.A, np.eye(self.n))) > 0:
            exit("Adjacency matrix has diagonal elements, meaning self loops which are not allowed")
        self.W = self.get_W()
            
    def get_W(self, type="perron"):
        if type == "eye":
            W = np.eye(self.n)
        if type == "perron":
            epsilon = 1/(self.get_maximum_degree() + 1)
            W = np.eye(self.n) - epsilon*self.get_L()
        elif type == "stochastic":
            A = nx.adjacency_matrix(self.G)
            degrees = np.sum(self.A,0)
            W = A*degrees[:,None]
        return W

    def get_neighbours(self):
        neighbours = {i:set() for i in range(self.n)}
        for i in range(self.n):
            neighbours_i = self.A[i,:]
            for j in range(self.n):
                if neighbours_i[j]:
                    neighbours[i].add(j)
        return neighbours

    def plot(self, G=None, colors = None, transmission = None, schedule=None, show_type=None, show=True):
        # only handles one slot
        if type(schedule)==type([]):
            schedule=schedule[0]
        if not G:
            G = self.G
        if show_type=="colors":
            color_lst = [colors[i] for i in range(self.n)]
            label_dict = {i:"{}:{}".format(i, colors[i]) for i in range(self.n)}
            nx.draw_networkx(G, node_color=color_lst, labels=label_dict)
        elif show_type=="transmission":
            node_colors = ["#7FFF00" if (i in schedule) else "#1f78b4" for i in range(self.n)]
            edge_colors = ["#7FFF00" if (transmission[i,j]+transmission[j,i]>0) else "#DC143C" for (i,j) in G.edges]
            nx.draw_networkx(G, node_color=node_colors, edge_color=edge_colors)
        elif show_type=="formal":
            linewidths=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
            width=[0.3 for edge in G.edges()]
            edge_styles=["-" for edge in G.edges()]
            nx.draw_networkx(G, node_color="white", edge_color="black", width=width, linewidths=linewidths, edgecolors= "black", font_size=12, font_family="serif", node_size=500, style=edge_styles)
        elif show_type=="formal_transmission":
            # https://stackoverflow.com/questions/20183433/available-font-family-entry-for-draw-networkx-labels
            linewidths=[3 if (i in schedule) else 1 for i in range(self.n)]
            width=[3 if (transmission[i,j]+transmission[j,i]>0) else 0.3 for (i,j) in G.edges]
            edge_styles=["-" for edge in G.edges()]
            nx.draw_networkx(G, node_color="white", edge_color="black", width=width, linewidths=linewidths, edgecolors= "black", font_size=12, font_family="serif", node_size=500, style=edge_styles)
        else:
            nx.draw_networkx(G)
        if show:
            plt.show()

    def get_degrees(self):
        return np.sum(self.A,0)

    def get_maximum_degree(self):
        D = np.diag(np.sum(self.A,0))
        max_D = np.max(D)
        return max_D

    def get_L(self):
        return nx.laplacian_matrix(self.G).todense()

    def color(self, G = None, order=None):
        if not G:
            G = self.G
        if not order:
            G_colored = nx.greedy_color(G, "random_sequential")
        else:
            G_colored = nx.greedy_color(G, order)
        return G_colored

    def get_secondary_neighbours_graph(self):
        new_A = copy.deepcopy(self.A)
        for i in range(self.n):
            for j in range(self.n):
                if self.A[i,j] == 1:
                    new_A[i,:] = new_A[i,:] + self.A[j, :]
                    new_A[i,i] = 0
        new_A[new_A>0] = 1
        new_G = nx.from_numpy_array(new_A)
        return new_G

class Policy(object):
    def __init__(self, graph:GraphHandler, policy:str, p=0.1, resource_strategy="uniform", verbose=True):
        self.n = graph.n
        self.p = p
        self.policy = policy
        self.graph = graph
        if policy == "random" and verbose:
            print("Random policy using p={}".format(p))
        elif policy in ["deterministic_random_choice", "deterministic_static"]:
            if verbose:
                print("Setting up deterministic scheduling...")
            wcnf_setup = self.get_wcnf()
            sols = self.solve_max_sat_wcnf(wcnf_setup[0], all_optimal=True)[0]
            self.schedule_arrays = []
            for sol in sols:
                node_idxs = self.get_node_idx_from_link_idxs(sol, wcnf_setup[1]) 
                self.schedule_arrays.append(self.get_schedule_array_from_node_idxs(node_idxs))
            if verbose:
                print("Done")
        elif policy in ["deterministic_time_dependent"]:
            self.link_weights = [1 for i in range(len(graph.G.edges())*2)]

    def schedule(self, S = 1):
        """
        Main function for scheduling from probabilities to outcome.
        """
        schedule_t = []
        for s in range(S):
            schedule_s = set()
            if self.policy == "all":
                schedule_array = np.ones(self.n)
            elif self.policy == "random":
                schedule_array = np.random.binomial(1, self.p, size=self.n)
            elif self.policy == "degree_random":
                schedule_array = np.ones(self.n)
                neighbours = self.graph.get_neighbours()
                for i in range(self.n):
                    d_i = len(neighbours[i])
                    schedule_array[i] = np.random.binomial(1, 1/(d_i+1), size=1)
            elif self.policy == "deterministic_random_choice":
                schedule_array = random.choice(self.schedule_arrays)
            elif self.policy == "deterministic_static":
                schedule_array = self.schedule_arrays[0]
            elif self.policy == "deterministic_time_dependent":
                print("solving...")
                wcnf_setup = self.get_wcnf(self.link_weights)
                sols = self.solve_max_sat_wcnf(wcnf_setup[0], all_optimal=True)[0]
                schedule_arrays = []
                for sol in sols:
                    node_idxs = self.get_node_idx_from_link_idxs(sol, wcnf_setup[1]) 
                    schedule_arrays.append(self.get_schedule_array_from_node_idxs(node_idxs))
                sol_choice = random.randint(0, len(sols)-1)
                sol_choice_links = sols[sol_choice]
                self.update_link_weights(sol_choice_links)
                schedule_array = schedule_arrays[sol_choice]
            for i in range(self.n):
                if schedule_array[i]:
                    schedule_s.add(i)
            schedule_t.append(schedule_s)
        return schedule_t

    def get_optimal_prob_uniform(self, trials=1000):
        """
        Calculates the optimal probability (binomial dist) to maximize the number of 
        expected successful links in a slot, all devices have same probability.
        """
        probs = np.arange(0,1,1/trials)
        num_expected_successful = np.asarray([self.get_E_N_suc(prob) for prob in probs])
        return probs[np.argmax(num_expected_successful)]
 
    def schedule_deterministic_optimal(self):
        wcnf_setup = self.get_wcnf()
        sol = self.solve_max_sat_wcnf(wcnf_setup[0])
        node_idxs = self.get_node_idx_from_link_idxs(sol[0][0], wcnf_setup[1]) 
        return self.get_schedule_array_from_node_idxs(node_idxs)

    def get_wcnf(self, weights=[]):
        """
        Sets up the DIMACS WCNF expression.
        https://pysathq.github.io/docs/html/api/formula.html#pysat.formula.CNFPlus        
        """
        wcnf = WCNF()
        neighbours = self.graph.get_neighbours()

        # Setup var idxs (Note that WCNF var idxs starts from 1 and not 0)
        idxs = {}
        l = 1
        for i in range(self.n):
            for j in neighbours[i]:
                idxs[self.n*i + j] = l
                l += 1
        
        # i to j meaning P_j_i
        for i in range(self.n):
            for j in neighbours[i]:
                p_j_i = idxs[self.n*i + j]
                for k in (neighbours[j]): # k to j peaning P_j_k
                    if k!=i:
                        p_j_k = idxs[self.n*k + j]
                        wcnf.append([-p_j_i, -p_j_k], weight=10000)
                    p_k_j = idxs[self.n*j + k]
                    wcnf.append([-p_j_i, -p_k_j], weight=10000)
        for idx in idxs:
            if weights:
                weight = weights[idxs[idx]-1]
            else:
                weight = 1
            wcnf.append([idxs[idx]], weight=weight)
        inverse_idxs = {idxs[key]: key for key in idxs}
        return wcnf, inverse_idxs, idxs

    def solve_max_sat_wcnf(self, wcnf:WCNF, all_optimal=False, solver='gc3'):
        rc2 = RC2(wcnf,solver=solver)#gc3, gc4, mc
        sols = {}
        mincost = self.graph.n*(self.graph.n-1)
        for  m in rc2.enumerate():
            if (all_optimal and rc2.cost <= mincost) or ((not all_optimal) and rc2.cost < mincost):
                mincost = rc2.cost
                if rc2.cost in sols:
                    sols[rc2.cost].append(m)
                else:
                    sols[rc2.cost] = [m]
            else:
                break    
        num_suc_links = len(self.graph.G.edges())*2 - mincost      
        return sols[min(sols.keys())], num_suc_links
    
    def get_node_idx_from_link_idxs(self, link_idxs:dict, inverse_link_idxs:dict):
        """
        From successful links (transmission i to j) with idx = num_nodes*i + j
        Calculate i for all successful links, to get what nodes are broadcasting.
        """
        node_idxs = []
        for link_idx in link_idxs:
            if link_idx > 0:
                i_idxs = inverse_link_idxs[link_idx]
                node_idx = int((i_idxs - i_idxs%self.n)/self.n)
                if not node_idx in node_idxs:
                    node_idxs.append(node_idx)
        return node_idxs

    def update_link_weights(self, scheduled_links:list):
        """
        Note that weights in WCNF must always be >0 to work
        as a weight. As default lowest I have picked 1.
        Note that WCNF link idxs start at 1.
        """
        def update(x):
            return x + 2
        for i in range(len(self.link_weights)):
            if i+1 in scheduled_links:
                self.link_weights[i] = 1
            else:
                self.link_weights[i] = update(self.link_weights[i])

    def get_schedule_array_from_node_idxs(self, node_idxs:list):
        schedule_array = np.zeros(self.n, dtype=int)
        schedule_array[node_idxs] = 1
        return schedule_array
        
    def get_expected_throughput(self, p:float, S:int, num_edge_normalized=True):
        neighbours = self.graph.get_neighbours()
        E = 0
        for i in range(self.n):
            d_i = len(neighbours[i])
            E += d_i*(1-np.power((1-p*np.power((1-p),d_i)),S))  # p^suc from i to j
        if num_edge_normalized:
            E = E/(2*len(self.graph.G.edges))
        return E
    
    def get_expected_throughput_nodewise(self, p:float, S:int, num_edge_normalized=True):
        neighbours = self.graph.get_neighbours()
        E = 0
        for i in range(self.n):
            d_i = len(neighbours[i])
            E += (1-np.power((1-p*np.power((1-p),d_i)),S))  # p^suc from i to j
        if num_edge_normalized:
            E = E/(len(self.graph.G.nodes))
        return E

    def get_optimal_prob_throughput_S(self, S:int, precision=0.001):
        dmax = np.max(self.graph.get_degrees())
        dmin = np.min(self.graph.get_degrees())
        probs = np.arange(1/(dmax+1)-precision,1/(dmin+1)+precision,precision)
        expected_throughput = np.zeros(len(probs))
        for i in range(len(probs)):
            expected_throughput[i] = self.get_expected_throughput(probs[i], S) # general
        return probs[np.argmax(expected_throughput)]
    
    def get_E_N_suc(self, p:float, num_edge_normalized=True):
        neighbours = self.graph.get_neighbours()
        E = 0
        for i in range(self.n):
            for j in neighbours[i]:
                E += p*np.power((1-p), len(neighbours[j])) # p^suc from i to j
        if num_edge_normalized:
            E = E/(2*len(self.graph.G.edges))
        return E

    def get_E_W_row_stoch(self, p:float, W:np.ndarray):
        W = copy.deepcopy(W)
        neighbours = self.graph.get_neighbours()
        for i in range(self.n):
            for j in neighbours[i]:
                W[i,j] = W[i,j]*p*np.power((1-p), len(neighbours[i]))  # W from j to i * p^suc from j to i
        return self.make_row_stochastic(W)

    def get_W_UU_frob_norm_sqrd(self, W:np.ndarray):
        u = np.ones((self.n,1))
        E_W_UU = W - u@np.transpose(u)/self.n
        E_W_UU_frob_norm = np.linalg.norm(E_W_UU)
        return np.power(E_W_UU_frob_norm, 2)
        
    def get_W_UU_2_norm_sqrd(self, W:np.ndarray):
        u = np.ones((self.n,1))
        E_W_UU = W - u@np.transpose(u)/self.n
        E_W_UU_2_norm = np.linalg.norm(E_W_UU, ord=2)
        return np.power(E_W_UU_2_norm, 2)

    def get_W_UU_spectral_radius(self, W:np.ndarray):
        u = np.ones((self.n,1))
        W_UU = W - u@np.transpose(u)/self.n
        return self.get_spectral_radius(W_UU)

    def get_spectral_radius(self, W:np.ndarray):   
        return np.max(np.abs(np.linalg.eig(W)[0]))

    def make_row_stochastic(self, W:np.ndarray):
        n = W.shape[0]
        I = np.eye(n)
        W_stoch = copy.deepcopy(W)
        W_stoch[I==1] = 1 - np.sum(np.multiply(W, np.ones((n,n))-I), axis=1)
        return W_stoch 

class CommModel(object):
    def __init__(self, graph:GraphHandler, comm="imperfect"):
        self.A = graph.A
        self.n = graph.n
        self.neighbours = graph.get_neighbours()
        self.comm = comm

    def broadcast(self, schedule:set, W:np.ndarray):
        """
        schedule: List of node idxs to be activated
        W: mixing matrix

        Note that in this implementation there is always self-transmission, meaning diagonal is eye.
        This is needed since W_t typically has self-transmission (non-zero diagonal), and that will never be
        interupted by any interference.
        """
        W_t = copy.deepcopy(W)
        if self.comm == "imperfect":
            transmission = np.zeros((self.n, self.n))
            K = len(schedule)
            for k in range(K):
                success_k = self.A + np.eye(self.n)
                for i in range(self.n):
                    for j in self.neighbours[i]:
                        interference = bool(len(schedule[k].intersection((self.neighbours[j].difference({i})).union({j}))))
                        if interference or (i not in schedule[k]):
                            success_k[j,i] = 0
                transmission += success_k
            transmission[transmission>0] = 1
        elif self.comm == "perfect":
            transmission = np.zeros((self.n, self.n))
            K = len(schedule)
            for k in range(K):
                success_k = self.A + np.eye(self.n)
                for i in range(self.n):
                    for j in self.neighbours[i]:
                        if i not in schedule[k]:
                            success_k[j,i] = 0
                transmission += success_k
            transmission[transmission>0] = 1
        else:
            exit("{} is an unknown communication model!".format(self.comm))

        # element-wise multiplication
        return np.multiply(W_t, transmission), transmission

class Resource(object):
    def __init__(self):
        pass
    
    def get(self, T:int, type="uniform", constraint=-1, a=3, b=0):
        if type=="delay_uniform":
            K = self.delay_uniform(T, constraint, a, b)
        elif type=="uniform_random":
            K = self.uniform_random(T, a, b)
        elif type=="nothing":
            K =  [0 for t in range(T)]
        elif type =="every_nth":
            K = self.every_nth(T, a, b)
        else:
            exit("Unknown resource allocation type: {}".format(type))

        # Limit to constraint
        if constraint > -1:
            left = constraint
            for i in range(len(K)):
                k = K[i]
                if k <= left and k > 0:
                    left -= k
                elif left == 0:
                    K[i] = 0
                elif k > 0:
                    K[i] = left
                    left = 0
        return K
    
    def every_nth(self, T:int, a:int, b:int):
        if b == 0:
            print("n cannot be 0")
            b = 1
        K = np.zeros(T,dtype=int).tolist()
        for t in range(T):
            if (t%b==0) and (t!=0):
                K[t] = a
        return K

    def delay_uniform(self, T:int, constraint:int, a:int, b:int):
        delay = b
        num_k = int(constraint/a)
        K = np.array([a for t in range(T)])
        K[0:delay] = 0
        K[delay+num_k+1:-1]=0
        return list(K)
    
    def uniform_random(self, T:int, a:int, b=0):
        delay = b
        K = list(np.random.randint(low=0, high=a+1, size=T))
        for i in range(delay):
            K[i] = 0
        return K
    
class DatasetSplit(Dataset):
    def __init__(self, dataset:Dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class DeviceHandler(object):
    def __init__(self, args:argparse.Namespace , data:DataHandler, model:ModelHandler, calculate_l2_norm=False, p_grad=1):
        self.devices = []
        self.model = model
        for i in range(args.num_users):
            local_data = DataLoader(DatasetSplit(data.train, data.idxs[i]), batch_size=args.local_bs, shuffle=True)
            device = Device(i, model.local_models[i], args.lr, args.loss_function, local_data, p_grad)
            device.calculate_l2_norm=calculate_l2_norm
            self.devices.append(device)
    
    def reset(self):
        for device in self.devices:
            device.reset()

class Device(object):
    def __init__(self, idx:int, model, lr:float, loss_function, data:DataLoader, p_grad:float):
        self.cache = []
        self.model = copy.deepcopy(model)
        self.model_copy = copy.deepcopy(model)
        self.data = data # dataloader
        self.dataiter = iter(self.data)

        self.idx = idx
        self.lr = lr
        self.loss_function = loss_function
        self.p_grad = p_grad
        self.calculate_l2_norm = False
        self.l2norms = []

        # For energy consumption tests
        self.battery_level = 1
        self.used_energy = 0
        self.E_comp = 2
        self.E_trans = 2

        self.grad_warning = False
        if p_grad != 1:
            print("WARNING, p_grad != 1")
        else:
            self.grad_warning = True
    
    def get_l2_norm_of_grad(self, grad:tuple):
        l2norm = 0
        for tensor in grad:
            l2norm += torchsum(torchpow(tensor,2))
        return torchpow(l2norm,1/2)

    def update(self, devices:list, W:np.ndarray, schedule:list):
        # Update model
        mix = self.mix(devices, W)

        # Randomly use gradient or not, for testing
        if not int(np.random.binomial(1, self.p_grad)):
            grad = 0
            if self.grad_warning:
                print("ERROR, GRAD IS RANDOM BUT SHOULD NOT BE")
        else:
            grad = self.grad()
            if self.calculate_l2_norm:
                self.l2norms.append(self.get_l2_norm_of_grad(grad))
        new_model_sd = self.step(mix, grad)
        self.model.load_state_dict(new_model_sd)

        # Update energy usage
        for slot_schedule in schedule:
            if self.idx in slot_schedule:
                self.used_energy += self.E_trans

    def add_state_dicts(self, a:dict, b:dict, c_a = 1, c_b = 1):
        """
        Adds two state dicts a + b from the same model architecture.
        c_a and c_b are coefficients if a different scaling is wanted.
        """
        c = copy.deepcopy(a)
        for k in a.keys():
            c[k] = c[k]*c_a + b[k]*c_b 
        return c

    def step(self, mix:dict, grad:tuple):
        """
        Difference between this function and add_state_dicts is that grad is not a state_dict.
        """
        mix = copy.deepcopy(mix)
        if grad:
            for i, k in enumerate(mix.keys()):
                mix[k] += -1*self.lr*grad[i]
        return mix

    def grad(self):
        """
        Calculates the gradient of a model and loss with respect to input data.
        WARNING: Always uses 1 batch.
        """
        self.used_energy += self.E_comp
        model = self.get_model_copy()
        model.train()

        # Take the next batch, if at the end reset the iterator.
        try:
            (data,target) = next(self.dataiter)
        except:
            self.dataiter = iter(self.data)
            (data,target) = next(self.dataiter)

        model.zero_grad()
        log_probs = model(data)
        loss = self.loss_function(log_probs, target)
        return grad(loss, model.parameters())

    def mix(self, devices: list, W: np.ndarray):
        """
        Calculates the averaging part of DSGD for a device i using its neighbours j and weights W.
        """
        update_weights = W[self.idx,:]
        avg_update = self.get_model_state_dict_copy()

        # Update based on itself
        for k in avg_update.keys():
            avg_update[k] = mul(avg_update[k], update_weights[self.idx])

        # Update based on neighbours
        for j in range(len(devices)):
            if j != self.idx:
                if update_weights[j]:
                    neighbour_model = devices[j].get_model_state_dict_copy()
                    avg_update = self.add_state_dicts(avg_update, neighbour_model, c_b = update_weights[j])

        return avg_update

    def get_model_copy(self):
        return copy.deepcopy(self.model)

    def get_model_state_dict_copy(self):
        return copy.deepcopy(self.model.state_dict())
    
    def reset(self):
        self.battery_level = 1
        self.used_energy = 0
        self.model.load_state_dict(self.model_copy.state_dict())
        self.l2norms = []
        self.dataiter = iter(self.data)

def dfl(W:np.ndarray, test_data, devices:DeviceHandler, policy:Policy, comm:CommModel, args:argparse.Namespace, S = [], reset_devices=True, auto_prob=False):    
    if reset_devices:
        print("Resetting devices in dfl()")
        for device in devices:
            device.reset()

    performance = []
    energies = []
    if not S:
        S = [1 for e in range(args.T)]
    for t in range(args.T): 
        if (t%args.status_freq == 0):
            performance.append([t, get_performance(devices, test_data, args)])
            energies.append([t, sum([device.used_energy for device in devices])/len(devices)])
            print("t={}/{}, accuracy={}, loss={}".format(t,args.T, performance[int(t/args.status_freq)][1][0], performance[int(t/args.status_freq)][1][1]))
        if auto_prob:
            policy.p = policy.get_optimal_prob_throughput_S(S[t])
        
        schedule = policy.schedule(S = S[t])
        W_t, transmission = comm.broadcast(schedule, W)
        W_t = make_row_stochastic(W_t)

        for device in devices:
            device.update(devices, W_t, schedule)

    performance.append([t, get_performance(devices, test_data, args)])
    energies.append([t, sum([device.used_energy for device in devices])/len(devices)])
    print("t={}/{}, accuracy={}, loss={}".format(args.T, args.T, performance[int(args.T/args.status_freq)][1][0], performance[int(args.T/args.status_freq)][1][1]))

    return {"performance": performance, "energies": energies}

def get_performance(devices:DeviceHandler, datatest, args:argparse.Namespace, verbose=False):
    """
    Get performance of list of models, meaning accuracy and loss over batches of test data.
    """
    accuracies = []
    losses = []
    N = len(devices)
    n = 1
    for device in devices:
        if verbose:
            print("device: ", n,N)
        n+=1
        model = device.model
        model.eval()
        test_loss = 0
        correct = 0
        data_loader = DataLoader(datatest, batch_size=args.bs, shuffle=True)

        num_batches = args.nb
        if (not num_batches) or (num_batches > len(data_loader)):
            num_batches = len(data_loader)
            if not num_batches:
                num_batches = 1
        i = 1
        for idx, (data, target) in enumerate(data_loader):
            if i > num_batches:
                break
            i += 1
            output = model(data)
            test_loss += args.loss_function(output, target).item()

            if args.dataset in ["epsilon", "mnist", "2clusters", "4clusters"]:
                y_pred = output.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
            elif args.dataset in ["line"]:
                correct+=tensor(0)
            else:
                exit("Unsupported dataset")
        accuracies.append((100.00 * correct / (num_batches*args.bs)).item())
        losses.append(test_loss/(num_batches))
    return np.average(accuracies), np.average(losses)

def parse_loss_function(model:str, dataset:str):
    if model == "mlp" and dataset in ["mnist"]:
        return nn.CrossEntropyLoss()
    if model == "cnn" and dataset in ["mnist"]:
        return nn.CrossEntropyLoss()
    if model in ["line", "linear_model"] and dataset == "line":
        return nn.MSELoss()
    if model in ["linear_model"] and dataset in ["2clusters", "4clusters", "epsilon"]:
        return nn.CrossEntropyLoss()
    exit("Unsupported model and dataset combination! Try following:\n*mlp with mnist\n*cnn with mnist\n*.inear with epsilon\n*line with line")

def make_row_stochastic(W:np.ndarray):
    n = W.shape[0]
    I = np.eye(n)
    W_stoch = copy.deepcopy(W)
    W_stoch[I==1] = 1 - np.sum(np.multiply(W, np.ones((n,n))-I), axis=1)
    return W_stoch 



import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
import numpy as np
import torch
import random

from options import args_parser

from dfl import *
import time
FONTNAME = "arial"

"""
Note: torch seed ensures that every run of this script is identical. 
However separate models will still have different parameters.
"""

def Fig_3_1():
    random.seed(3)
    np.random.seed(1)

    graph = GraphHandler(7, topology="erdos_random",p=0.32)
    comm = CommModel(graph, "perfect")

    fig, ax = plt.subplots(1,3)
    # Schedule 1
    schedule = [{1}]
    W, T = comm.broadcast(schedule, graph.W)
    linewidths=[3 if (i in schedule[0]) else 1 for i in range(graph.n)]
    width=[3 if (T[i,j]+T[j,i]>0) else 0.3 for (i,j) in graph.G.edges()]
    edge_styles=["-" for edge in graph.G.edges()]
    nx.draw_networkx(graph.G, ax=ax[0],node_color="white", edge_color="black", width=width, linewidths=linewidths, edgecolors= "black", font_size=12, font_family="serif", node_size=500, style=edge_styles)
    random.seed(3)
    np.random.seed(1)

    # Schedule 2
    schedule = [{1,0}]
    W, T = comm.broadcast(schedule, graph.W)
    linewidths=[3 if (i in schedule[0]) else 1 for i in range(graph.n)]
    width=[3 if (T[i,j]+T[j,i]>0) else 0.3 for (i,j) in graph.G.edges()]
    edge_styles=["-" for edge in graph.G.edges()]
    nx.draw_networkx(graph.G, ax=ax[1],node_color="white", edge_color="black", width=width, linewidths=linewidths, edgecolors= "black", font_size=12, font_family="serif", node_size=500, style=edge_styles)
    random.seed(3)
    np.random.seed(1)

    # Schedule 3
    schedule = [{1,6}]
    W, T = comm.broadcast(schedule, graph.W)
    linewidths=[3 if (i in schedule[0]) else 1 for i in range(graph.n)]
    width=[3 if (T[i,j]+T[j,i]>0) else 0.3 for (i,j) in graph.G.edges()]
    edge_styles=["-" for edge in graph.G.edges()]
    nx.draw_networkx(graph.G, ax=ax[2],node_color="white", edge_color="black", width=width, linewidths=linewidths, edgecolors= "black", font_size=12, font_family="serif", node_size=500, style=edge_styles)
    
    hfont2= {'fontname':'serif','fontsize':16,'fontstyle':'normal'}
    ax[0].set_title("Node 1 broadcasts", **hfont2)
    ax[1].set_title("Node 1 and 0 broadcasts", **hfont2)
    ax[2].set_title("Node 1 and 6 broadcasts", **hfont2)
    plt.show()

def Fig_3_2():
    random.seed(3)
    np.random.seed(1)

    graph = GraphHandler(7, topology="erdos_random",p=0.32)
    comm = CommModel(graph, "imperfect")

    fig, ax = plt.subplots(1,3)
    # Schedule 1
    schedule = [{1}]
    W, T = comm.broadcast(schedule, graph.W)
    linewidths=[3 if (i in schedule[0]) else 1 for i in range(graph.n)]
    width=[3 if (T[i,j]+T[j,i]>0) else 0.3 for (i,j) in graph.G.edges()]
    edge_styles=["-" for edge in graph.G.edges()]
    nx.draw_networkx(graph.G, ax=ax[0],node_color="white", edge_color="black", width=width, linewidths=linewidths, edgecolors= "black", font_size=12, font_family="serif", node_size=500, style=edge_styles)
    random.seed(3)
    np.random.seed(1)

    # Schedule 2
    schedule = [{1,0}]
    W, T = comm.broadcast(schedule, graph.W)
    linewidths=[3 if (i in schedule[0]) else 1 for i in range(graph.n)]
    width=[3 if (T[i,j]+T[j,i]>0) else 0.3 for (i,j) in graph.G.edges()]
    edge_styles=["-" for edge in graph.G.edges()]
    nx.draw_networkx(graph.G, ax=ax[1],node_color="white", edge_color="black", width=width, linewidths=linewidths, edgecolors= "black", font_size=12, font_family="serif", node_size=500, style=edge_styles)
    random.seed(3)
    np.random.seed(1)

    # Schedule 3
    schedule = [{1,6}]
    W, T = comm.broadcast(schedule, graph.W)
    linewidths=[3 if (i in schedule[0]) else 1 for i in range(graph.n)]
    width=[3 if (T[i,j]+T[j,i]>0) else 0.3 for (i,j) in graph.G.edges()]
    edge_styles=["-" for edge in graph.G.edges()]
    nx.draw_networkx(graph.G, ax=ax[2],node_color="white", edge_color="black", width=width, linewidths=linewidths, edgecolors= "black", font_size=12, font_family="serif", node_size=500, style=edge_styles)
    
    hfont1= {'fontname':'serif','fontsize':14,'fontstyle':'normal'}
    hfont2= {'fontname':'serif','fontsize':16,'fontstyle':'normal'}

    ax[0].set_title("Node 1 broadcasts", **hfont2)
    ax[1].set_title("Node 1 and 0 broadcasts", **hfont2)
    ax[2].set_title("Node 1 and 6 broadcasts", **hfont2)
    plt.show()

def Fig_3_3_c_E12():
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(1)
    random.seed(4)
    np.random.seed(1)

    args.num_users = 20
    args.policy = "random"
    args.model = "line"
    args.dataset = "line"
    args.topology = "ring"
    args.comm = "imperfect"

    # setup
    graph = GraphHandler(args.num_users, topology=args.topology)   
    comm = CommModel(graph, args.comm)
    policy = Policy(graph, policy=args.policy)

    # verify setup
    print(args)
    graph.plot()    
    probs = np.arange(0,1,0.001)

    num_trials = 250
    num_successful = np.zeros((len(probs), num_trials))
    num_expected_successful = np.zeros((len(probs)))
    for i in range(len(probs)):
        num_expected_successful[i] = policy.get_E_N_suc(probs[i]) # general
        policy = Policy(graph, policy=args.policy, p=probs[i], verbose=False)
        print("{}/{}".format(i,len(probs)))
        for j in range(num_trials):
            schedule = policy.schedule()
            W_t, transmission = comm.broadcast(schedule, graph.W)
            num_successful[i,j] = int(np.sum(transmission) - args.num_users)
    average_num_successful = np.average(num_successful, axis=1)/(2*len(graph.G.edges))
    
    # optional save
    if 0:
        np.save("experimentdata/E12.npy",np.array([probs,average_num_successful,num_expected_successful]))

    # plot
    plt.plot(probs, average_num_successful, label="Average")
    plt.plot(probs, num_expected_successful, label="Expected")
    plt.title("E12 ring 20 devices average and expected num suc links")
    plt.xlabel("p")
    plt.ylabel("ratio link success")
    plt.legend()
    plt.show()

def Fig_3_3_d_E13():
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(1)
    random.seed(4)
    np.random.seed(1)

    args.num_users = 20
    args.policy = "random"
    args.model = "line"
    args.dataset = "line"
    args.topology = "complete"
    args.comm = "imperfect"

    # setup
    graph = GraphHandler(args.num_users, topology=args.topology)   
    comm = CommModel(graph, args.comm)
    policy = Policy(graph, policy=args.policy)

    # verify setup
    print(args)
    graph.plot()    
    probs = np.arange(0,1,0.001)

    num_trials = 250
    num_successful = np.zeros((len(probs), num_trials))
    num_expected_successful = np.zeros((len(probs)))
    for i in range(len(probs)):
        num_expected_successful[i] = policy.get_E_N_suc(probs[i]) # general
        policy = Policy(graph, policy=args.policy, p=probs[i], verbose=False)
        print("{}/{}".format(i,len(probs)))
        for j in range(num_trials):
            schedule = policy.schedule()
            W_t, transmission = comm.broadcast(schedule, graph.W)
            num_successful[i,j] = int(np.sum(transmission) - args.num_users)

    average_num_successful = np.average(num_successful, axis=1)/(2*len(graph.G.edges))
    
    # optional save
    if 0:
        np.save("experimentdata/E13.npy",np.array([probs,average_num_successful,num_expected_successful]))

    # plot
    plt.plot(probs, average_num_successful, label="Average")
    plt.plot(probs, num_expected_successful, label="Expected")
    plt.title("E13 complete 20 devices average and expected num suc links")
    plt.xlabel("p")
    plt.ylabel("ratio link success")
    plt.legend()
    plt.show()

def Fig_3_4_a_PE5_1():
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(1)
    random.seed(4)
    np.random.seed(1)

    args.nb=20
    args.lr=0.01
    args.num_users = 20
    args.policy = "random"
    args.model = "line"
    args.dataset = "line"
    args.topology = "erdos_random"
    args.comm = "imperfect"

    # setup
    graph = GraphHandler(args.num_users, topology=args.topology)   
    policy = Policy(graph, policy=args.policy)

    # verify setup
    print(args)
    graph.plot()    
    probs = np.arange(0,1,0.001)
    num_expected_successful = np.asarray([policy.get_E_N_suc(prob) for prob in probs])
    E_W_UU_spectral_radius = np.asarray([policy.get_W_UU_spectral_radius(policy.get_E_W_row_stoch(prob, graph.W)) for prob in probs])
    num_expected_successful = (num_expected_successful-np.min(num_expected_successful))/(np.max(num_expected_successful)-np.min(num_expected_successful))
    E_W_UU_spectral_radius = (E_W_UU_spectral_radius-np.min(E_W_UU_spectral_radius))/(np.max(E_W_UU_spectral_radius)-np.min(E_W_UU_spectral_radius))
    
    argmax_num_E_suc = probs[np.argmax(num_expected_successful)]
    argmax_num_E_W_UU_spectral_radius = probs[np.argmin(E_W_UU_spectral_radius)]

    # optional save
    if 0:
        np.save("experimentdata/PE5.npy",np.array([argmax_num_E_suc,argmax_num_E_W_UU_spectral_radius]))

    # plot
    hfont1= {'fontname':FONTNAME,'fontsize':20,'fontstyle':'normal'}
    hfont2= {'fontname':FONTNAME,'fontsize':24,'fontstyle':'normal'}
    plt.rcParams["figure.figsize"] = (10,6)
    plt.plot(probs, E_W_UU_spectral_radius, "--", label=r"$\rho(\mathbb{E}[\overline{W}]-uu^T/n)$", color="black")
    plt.plot(probs, num_expected_successful, "-", label=r"$\mathbb{E}[N^{suc}]$", color="black")
    plt.vlines(x=argmax_num_E_W_UU_spectral_radius, ymin=-2, ymax=2, linewidth=0.5, color='gray')
    plt.vlines(x=argmax_num_E_suc, ymin=-2, ymax=2, linewidth=0.5, color='gray')
    plt.text(argmax_num_E_W_UU_spectral_radius+0.01, 0.4, r"$\rho(\mathbb{E}[\overline{W}]-uu^T/n)$" + "\n" + "{:.3f}".format(argmax_num_E_W_UU_spectral_radius),**hfont1)
    plt.text(argmax_num_E_suc-0.1, 0.6, r"$\mathbb{E}[N^{suc}]$" + "\n" + "{:.3f}".format(argmax_num_E_suc),**hfont1)
    plt.ylim(0,1)
    plt.xlim(0,1)    
    plt.xlabel("p, precision={}".format(1/len(probs)), **hfont2)
    plt.ylabel("val", **hfont2)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.show()

def Fig_3_4_c_PE6_1():
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(1)
    random.seed(4)
    np.random.seed(1)

    args.nb=20
    args.lr=0.01
    args.num_users = 20
    args.policy = "random"
    args.model = "line"
    args.dataset = "line"
    args.topology = "ring"
    args.comm = "imperfect"

    # setup
    graph = GraphHandler(args.num_users, topology=args.topology)   
    policy = Policy(graph, policy=args.policy)

    # verify setup
    print(args)
    graph.plot()    
    probs = np.arange(0,1,0.001)
    num_expected_successful = np.asarray([policy.get_E_N_suc(prob) for prob in probs])
    E_W_UU_spectral_radius = np.asarray([policy.get_W_UU_spectral_radius(policy.get_E_W_row_stoch(prob, graph.W)) for prob in probs])
    num_expected_successful = (num_expected_successful-np.min(num_expected_successful))/(np.max(num_expected_successful)-np.min(num_expected_successful))
    E_W_UU_spectral_radius = (E_W_UU_spectral_radius-np.min(E_W_UU_spectral_radius))/(np.max(E_W_UU_spectral_radius)-np.min(E_W_UU_spectral_radius))
    
    argmax_num_E_suc = probs[np.argmax(num_expected_successful)]
    argmax_num_E_W_UU_spectral_radius = probs[np.argmin(E_W_UU_spectral_radius)]

    # optional save
    if 0:
        np.save("experimentdata/PE6.npy",np.array([argmax_num_E_suc,argmax_num_E_W_UU_spectral_radius]))
    # plot
    hfont2= {'fontname':FONTNAME,'fontsize':24,'fontstyle':'normal'}
    plt.rcParams["figure.figsize"] = (10,6)
    plt.plot(probs, E_W_UU_spectral_radius, "--", label=r"$\rho(\mathbb{E}[\overline{W}]-uu^T/n)$", color="black")
    plt.plot(probs, num_expected_successful, "-", label=r"$\mathbb{E}[N^{suc}]$", color="black")
    plt.vlines(x=argmax_num_E_W_UU_spectral_radius, ymin=-2, ymax=2, linewidth=0.5, color='gray')
    plt.vlines(x=argmax_num_E_suc, ymin=-2, ymax=2, linewidth=0.5, color='gray')
    plt.text(argmax_num_E_W_UU_spectral_radius+0.01, 0.4, r"$\rho(\mathbb{E}[\overline{W}]-uu^T/n)$" + "\n" + "{:.3f}".format(argmax_num_E_W_UU_spectral_radius), **hfont2)
    plt.text(argmax_num_E_suc-0.13, 0.6, r"$\mathbb{E}[N^{suc}]$" + "\n" + "{:.3f}".format(argmax_num_E_suc), **hfont2)
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.xlabel("p, precision={}".format(1/len(probs)), **hfont2)
    plt.ylabel("val", **hfont2)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.show()

def Fig_3_5_a_E21_1():
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(1)
    random.seed(4)
    np.random.seed(1)

    args.nb=20
    args.lr=0.01
    args.num_users = 20
    args.policy = "random"
    args.model = "line"
    args.dataset = "line"
    args.topology = "erdos_random"
    args.comm = "imperfect"

    # setup
    graph = GraphHandler(args.num_users, topology=args.topology)   
    comm = CommModel(graph, args.comm)
    policy = Policy(graph, policy=args.policy)

    # verify setup
    print(args)
    graph.plot()    
    precision = 0.001
    probs = np.arange(0,1,precision)
    S = [1,3,5,10,20,50,500]
    expected_throughput = np.zeros((len(probs),len(S)))

    for i in range(len(probs)):
        if (i%100 == 0):
            print("evaluating expected",i,len(probs))
        for idx, s in enumerate(S):
            expected_throughput[i,idx] = policy.get_expected_throughput(probs[i],s) # general

    # optional save
    if 0:
        np.save("experimentdata/E21_1.npy",np.array([expected_throughput]))
    
    # plot
    hfont2= {'fontname':FONTNAME,'fontsize':18,'fontstyle':'normal'}
    fig, ax = plt.subplots(1)
    plt.plot(probs, expected_throughput[:,6],":", label="S={}, p*={:.3f}".format(100, probs[np.argmax(expected_throughput[:,6])]), color="black")
    plt.plot(probs, expected_throughput[:,5], label="S={}, p*={:.3f}".format(50, probs[np.argmax(expected_throughput[:,5])]), color="black")
    plt.plot(probs, expected_throughput[:,4], label="S={}, p*={:.3f}".format(20, probs[np.argmax(expected_throughput[:,4])]), color="black")
    plt.plot(probs, expected_throughput[:,3], label="S={}, p*={:.3f}".format(10, probs[np.argmax(expected_throughput[:,3])]), color="black")
    plt.plot(probs, expected_throughput[:,2], label="S={}, p*={:.3f}".format(5, probs[np.argmax(expected_throughput[:,2])]), color="black")
    plt.plot(probs, expected_throughput[:,1], label="S={}, p*={:.3f}".format(3, probs[np.argmax(expected_throughput[:,1])]), color="black")
    plt.plot(probs, expected_throughput[:,0],"--", label="S={}, p*={:.3f}".format(1, probs[np.argmax(expected_throughput[:,0])]), color="black")

    plt.xlabel("Access probability, precision={}".format(precision), **hfont2)
    plt.ylabel(r"$\mathbb{E}[\overline{Q}]$", **hfont2)
    plt.legend(fontsize=14)
    ax.set_xticks([0.0,0.2,0.4,0.6,0.8,1.0], [0.0,0.2,0.4,0.6,0.8,1.0], fontsize=13)
    ax.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0], [0.0,0.2,0.4,0.6,0.8,1.0], fontsize=13)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1.1)
    plt.show()

def Fig_3_6_E51():
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(1)
    random.seed(4)
    np.random.seed(1)
    args.num_users = 20
    args.topology = "erdos_random"

    # setup
    graph = GraphHandler(args.num_users, topology=args.topology)   

    # verify setup
    print(args)
    graph.plot()    
    precision = 0.001

    D = np.arange(1,101,1)
    P = np.arange(0,1+1*precision,precision)
    P_suc = np.zeros((len(D),len(P)))
    S = 10
    for i in range(len(P)):
        print(i/len(P))
        p = P[i]
        for d in D:
            P_suc[d-1,i] = np.log((1-np.power((1-p*np.power((1-p),d)),S)))
    
    # plot
    hfont2= {'fontname':FONTNAME,'fontsize':18,'fontstyle':'normal'}
    plt.rcParams['contour.negative_linestyle'] = 'solid'
    fig, ax = plt.subplots()
    CS = ax.contour(P, D, P_suc, 7, colors='k')        
    manual_locations = [
            (0.2, 25), (0.3, 30),(0.4, 35), (0.4, 40),(0.5, 40),(0.6, 60)]
    ax.set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],fontsize=14)
    ax.set_yticks([1,10,20,30,40,50,60,70,80,90,99], [1,10,20,30,40,50,60,70,80,90,100],fontsize=14)
    ax.set_xlim(0,1.01)
    ax.clabel(CS, inline=True, fontsize=10, manual=manual_locations)
    ax.set_xlabel("Probability p", **hfont2)
    ax.set_ylabel("Degree d", **hfont2)
    plt.show()

def Fig_4_1():
    torch.manual_seed(1)
    random.seed(4)
    np.random.seed(1)
    graph = GraphHandler(20, topology="erdos_random")
    print(np.mean(graph.get_degrees()), np.max(graph.get_degrees()), np.min(graph.get_degrees()))
    print(graph.G.edges)
    graph.plot(show_type="formal", show=False)
    plt.show()

def Fig_4_2_b():
    random.seed(3)
    np.random.seed(7)
    num_users = 20
    iid = True
    dataset = "4clusters"
    data = DataHandler(dataset, num_users, iid)
    data.train.plot(show=False)
    hfont1= {'fontname':'serif','fontsize':14,'fontstyle':'normal'}
    hfont2= {'fontname':'serif','fontsize':18,'fontstyle':'normal'}
    plt.xlabel(r'$x_1$', **hfont2)
    plt.ylabel(r'$x_2$', **hfont2)
    plt.yticks([-1.5,0,1.5], [-1.5,0,1.5], fontsize=13)
    plt.xticks([-1.5,0,1.5], [-1.5,0,1.5], fontsize=13)
    plt.xlim(-1.7,1.7)
    plt.ylim(-1.7,1.7)
    plt.legend(fontsize=14)
    plt.show()

def Fig_4_2_a():
    random.seed(3)
    np.random.seed(7)
    num_users = 20
    iid = True
    dataset = "line"
    data = DataHandler(dataset, num_users, iid)
    data.train.plot(show=False)
    hfont1= {'fontname':'serif','fontsize':14,'fontstyle':'normal'}
    hfont2= {'fontname':'serif','fontsize':18,'fontstyle':'normal'}
    plt.xlabel(r"$x$", **hfont2)
    plt.ylabel(r"$y$", **hfont2)
    plt.yticks([-2,0,2], [-2,0,2], fontsize=13)
    plt.xticks([0,0.5,1], [0,0.5,1], fontsize=13)
    plt.xlim(-0.1,1.1)
    plt.ylim(-3,3)
    plt.show()

def Fig_4_3_a_E2():
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(1)
    random.seed(4)
    np.random.seed(1)

    args.nb=20
    args.lr=0.01
    args.num_users = 20
    args.policy = "random"
    args.model = "line"
    args.dataset = "line"
    args.topology = "ring"
    args.comm = "perfect"
    args.iid = False
    args.loss_function = parse_loss_function(args.model, args.dataset)

    # setup
    graph = GraphHandler(args.num_users, topology=args.topology)   
    comm = CommModel(graph, args.comm)
    data = DataHandler(args.dataset, args.num_users, args.iid)
    model = ModelHandler(args, data)
    devices = DeviceHandler(args, data, model)

    # verify setup
    print(args)
    graph.plot()
    data.train.plot()
    data.test.plot()

    probs = [1]
    losses = []
    accuracies = []
    times = []

    time1 = time.time()
    for prob in probs:
        policy = Policy(graph, policy=args.policy, p=prob)
        performance = dfl(graph.W, data.test, devices.devices, policy, comm, args)["performance"]
        times.append([p[0] for p in performance])
        losses.append([p[1][-1] for p in performance])
        accuracies.append([p[1][0] for p in performance])
    print("time (S): ", time.time() - time1)
    
    # optional save
    if 0:
        np.save("experimentdata/E2.npy",np.array([times,losses,accuracies]))
    # plotting
    for i in range(len(losses)):
        plt.plot(times[i], losses[i])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend(probs)
    name = "E2 line line p=1 ring interference free, non-iid"
    plt.title(name)
    plt.grid(True)
    plt.show()

def Fig_4_3_b_E3():
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(1)
    random.seed(4)
    np.random.seed(1)

    args.T = 5000
    args.status_freq = 100
    args.nb=20
    args.local_bs=100
    args.lr=0.5
    args.num_users = 20
    args.policy = "random"
    args.model = "linear_model"
    args.dataset = "4clusters"
    args.topology = "ring"
    args.comm = "perfect"
    args.iid = False
    args.loss_function = parse_loss_function(args.model, args.dataset)

    # setup
    graph = GraphHandler(args.num_users, topology=args.topology)   
    comm = CommModel(graph, args.comm)
    data = DataHandler(args.dataset, args.num_users, args.iid)
    model = ModelHandler(args, data)
    devices = DeviceHandler(args, data, model)

    # verify setup
    print(args)
    graph.plot()
    data.train.plot()
    data.test.plot()

    probs = [1]
    losses = []
    accuracies = []
    times = []

    time1 = time.time()
    for prob in probs:
        policy = Policy(graph, policy=args.policy, p=prob)
        performance = dfl(graph.W, data.test, devices.devices, policy, comm, args)["performance"]
        times.append([p[0] for p in performance])
        losses.append([p[1][-1] for p in performance])
        accuracies.append([p[1][0] for p in performance])
    print("time (S): ", time.time() - time1)
    
    # optional save
    if 0:
        np.save("experimentdata/E3.npy",np.array([times,losses,accuracies]))
    # plot
    for i in range(len(losses)):
        plt.plot(times[i], losses[i])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend(probs)
    name = "E3 linear model 4clusters p=1 ring interference free, non-iid"
    plt.title(name)
    plt.grid(True)
    plt.show()

def Fig_4_4_a_E4_1():
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(1)
    random.seed(4)
    np.random.seed(1)

    args.status_freq = 2
    args.nb=20
    args.lr=0.01
    args.num_users = 20
    args.policy = "random"
    args.model = "line"
    args.dataset = "line"
    args.topology = "erdos_random"
    args.comm = "imperfect"
    args.iid = False
    args.loss_function = parse_loss_function(args.model, args.dataset)
    num_trials = 10

    # setup
    graph = GraphHandler(args.num_users, topology=args.topology)   
    comm = CommModel(graph, args.comm)
    data = DataHandler(args.dataset, args.num_users, args.iid)
    model = ModelHandler(args, data)
    devices = DeviceHandler(args, data, model)

    # verify setup
    print(args)
    graph.plot()
    data.train.plot()
    data.test.plot()

    probs = [0,0.25,0.5,0.75,1]
    losses = []
    accuracies = []
    times = []
    for prob in probs:
        prob_losses = []
        prob_accuracies = []
        for t in range(num_trials):
            policy = Policy(graph, policy=args.policy, p=prob)
            performance = dfl(graph.W, data.test, devices.devices, policy, comm, args)["performance"]
            prob_losses.append([p[1][-1] for p in performance])
            prob_accuracies.append([p[1][0] for p in performance])
        losses.append(np.mean(np.array(prob_losses), axis=0))
        accuracies.append(np.mean(np.array(prob_accuracies), axis=0))
        times.append([p[0] for p in performance])

    # optional save
    if 0:
        np.save("experimentdata/E4_1.npy",np.array([times,losses,accuracies]))
    # plot
    for i in range(len(losses)):
        plt.plot(times[i], losses[i])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend(probs)
    name = "E4.1 line line various p erdos random with interference, non-iid"
    plt.title(name)
    plt.grid(True)
    plt.show()

def Fig_4_5_a_E5_1():
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(1)
    random.seed(4)
    np.random.seed(1)

    num_trials = 10
    args.T = 5000
    args.status_freq = 100
    args.nb=20
    args.lr=0.01
    args.num_users = 20
    args.policy = "random"
    args.model = "linear_model"
    args.dataset = "4clusters"
    args.topology = "erdos_random"
    args.comm = "imperfect"
    args.iid = False
    args.loss_function = parse_loss_function(args.model, args.dataset)
    
    # setup
    graph = GraphHandler(args.num_users, topology=args.topology)   
    comm = CommModel(graph, args.comm)
    data = DataHandler(args.dataset, args.num_users, args.iid)
    model = ModelHandler(args, data)
    devices = DeviceHandler(args, data, model)

    # verify setup
    print(args)
    graph.plot()
    data.train.plot()
    data.test.plot()

    probs = [0,0.25,0.5,0.75]
    losses = []
    accuracies = []
    times = []
    for prob in probs:
        prob_losses = []
        prob_accuracies = []
        for t in range(num_trials):
            policy = Policy(graph, policy=args.policy, p=prob)
            performance = dfl(graph.W, data.test, devices.devices, policy, comm, args)["performance"]
            prob_losses.append([p[1][-1] for p in performance])
            prob_accuracies.append([p[1][0] for p in performance])
        losses.append(np.mean(np.array(prob_losses), axis=0))
        accuracies.append(np.mean(np.array(prob_accuracies), axis=0))
        times.append([p[0] for p in performance])

    # optional save
    if 0:
        np.save("experimentdata/E5_1.npy",np.array([times,losses,accuracies]))

    # plot
    for i in range(len(losses)):
        plt.plot(times[i], losses[i])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend(probs)
    plt.grid(True)
    plt.show()

def Fig_4_6_aE9_1():
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(1)
    random.seed(4)
    np.random.seed(1)

    num_trials = 10
    args.status_freq = 2
    args.T = 100
    args.nb=20
    args.lr=0.01
    args.num_users = 20
    args.policy = "random"
    args.model = "line"
    args.dataset = "line"
    args.topology = "erdos_random"
    args.comm = "perfect"
    args.iid = True
    args.loss_function = parse_loss_function(args.model, args.dataset)

    # setup
    graph = GraphHandler(args.num_users, topology=args.topology)   
    comm = CommModel(graph, args.comm)
    data = DataHandler(args.dataset, args.num_users, args.iid)
    model = ModelHandler(args, data)
    devices = DeviceHandler(args, data, model)

    # verify setup
    print(args)
    graph.plot()
    data.train.plot()
    data.test.plot()

    probs = [0,0.25,0.5,0.75,1]
    losses = []
    accuracies = []
    times = []
    for prob in probs:
        prob_losses = []
        prob_accuracies = []
        for t in range(num_trials):
            print(t+1,num_trials)
            policy = Policy(graph, policy=args.policy, p=prob)
            performance = dfl(graph.W, data.test, devices.devices, policy, comm, args)["performance"]
            prob_losses.append([p[1][-1] for p in performance])
            prob_accuracies.append([p[1][0] for p in performance])
        losses.append(np.mean(np.array(prob_losses), axis=0))
        accuracies.append(np.mean(np.array(prob_accuracies), axis=0))
        times.append([p[0] for p in performance])

    # optional save
    if 0:
        np.save("experimentdata/E9_1.npy",np.array([times,losses,accuracies]))
        
    # plot
    for i in range(len(losses)):
        plt.plot(times[i], losses[i])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend(probs)
    name = "E9.1 IID line line IF various p erdos random"
    plt.title(name)
    plt.grid(True)
    plt.show()

def Fig_4_7_a_E14_1(save=False):
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(1)
    random.seed(4)
    np.random.seed(1)

    args.status_freq = 2
    args.nb=20
    args.lr=0.01
    args.num_users = 20
    args.policy = "random"
    args.model = "line"
    args.dataset = "line"
    args.topology = "erdos_random"
    args.comm = "imperfect"
    args.iid = False
    args.loss_function = parse_loss_function(args.model, args.dataset)
    num_trials = 10

    # setup
    graph = GraphHandler(args.num_users, topology=args.topology)   
    comm = CommModel(graph, args.comm)
    data = DataHandler(args.dataset, args.num_users, args.iid)
    model = ModelHandler(args, data)
    devices = DeviceHandler(args, data, model)
    policy = Policy(graph, policy=args.policy)

    # verify setup
    print(args)
    graph.plot()
    data.train.plot()
    data.test.plot()

    # calculate optimal probability wrt expected num links
    probs = np.arange(0,1,0.001)
    num_expected_successful = np.zeros((len(probs)))
    for i in range(len(probs)):
        num_expected_successful[i] = policy.get_E_N_suc(probs[i])
    optimal_p_1 = probs[np.argmax(num_expected_successful)]

    # setup probabilities
    probs = [optimal_p_1, 0.25, 0.1]
    print(probs)

    # run experiment
    losses = []
    times = []
    for prob in probs:
        prob_losses = []
        for t in range(num_trials):
            policy = Policy(graph, policy=args.policy, p=prob)
            performance = dfl(graph.W, data.test, devices.devices, policy, comm, args)["performance"]
            prob_losses.append([p[1][-1] for p in performance])
        times.append([p[0] for p in performance])
        losses.append(np.mean(np.array(prob_losses), axis=0))
    # optional save
    if 0:
        np.save("experimentdata/E14_1.npy",np.array([times,losses]))
    # plot
    for i in range(len(losses)):
        plt.plot(times[i], losses[i])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend(probs)
    name = "E14.1 erdos line line CD random policy, 10 trials"
    plt.title(name)
    plt.grid(True)
    plt.show()

def Fig_4_9_E18():
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(1)
    random.seed(4)
    np.random.seed(1)

    args.status_freq = 2
    args.nb=20
    args.lr=0.01
    args.num_users = 20
    args.policy = "random"
    args.model = "line"
    args.dataset = "line"
    args.topology = "erdos_random"
    args.comm = "imperfect"
    args.iid = False
    args.loss_function = parse_loss_function(args.model, args.dataset)

    # setup
    graph = GraphHandler(args.num_users, topology=args.topology)   
    comm = CommModel(graph, args.comm)
    data = DataHandler(args.dataset, args.num_users, args.iid)
    model = ModelHandler(args, data)
    devices = DeviceHandler(args, data, model)
    policy = Policy(graph, policy=args.policy)

    # verify setup
    print(args)
    graph.plot()
    data.train.plot()
    data.test.plot()

    optimal_p = policy.get_optimal_prob_uniform()

    times = []  
    T = 10
    # run for deterministic optimal w. random choice
    losses = []
    for t in range(T):
        print("deterministic random choice", t)
        policy = Policy(graph, policy="deterministic_random_choice")
        performance = dfl(graph.W, data.test, devices.devices, policy, comm, args)["performance"]
        times.append([p[0] for p in performance])
        losses.append([p[1][-1] for p in performance])
    losses_np_1 = np.mean(np.asarray(losses), axis=0)

    # run for deterministic optimal w. random choice
    losses = []
    for t in range(T):
        print("deterministic static", t)
        policy = Policy(graph, policy="deterministic_static")
        performance = dfl(graph.W, data.test, devices.devices, policy, comm, args)["performance"]
        times.append([p[0] for p in performance])
        losses.append([p[1][-1] for p in performance])
    losses_np_2 = np.mean(np.asarray(losses), axis=0)

    # run for random optimal
    losses = []
    for t in range(T):
        print("random policy", t)
        policy = Policy(graph, policy="random", p=optimal_p)
        performance = dfl(graph.W, data.test, devices.devices, policy, comm, args)["performance"]
        times.append([p[0] for p in performance])
        losses.append([p[1][-1] for p in performance])
    losses_np_3 = np.mean(np.asarray(losses), axis=0)

    # optional save
    if 0:
        np.save("experimentdata/E18.npy",np.array([times[0],losses_np_1, losses_np_2, losses_np_3]))
    # plot
    plt.plot(times[0], losses_np_3, "-*", label="random: {}".format(optimal_p))
    plt.plot(times[0], losses_np_1, "-o", label="deterministic random")
    plt.plot(times[0], losses_np_2, "-x", label="deterministic static")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    name = "E18 random vs deterministic optimal, erdos, {} trials".format(T)
    plt.title(name)
    plt.grid(True)
    plt.show()

def Fig_4_10_a_E49_1(save=False):
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(1)
    random.seed(4)
    np.random.seed(1)

    args.status_freq = 5
    args.nb=20
    args.lr=0.01
    args.num_users = 20
    args.policy = "random"
    args.model = "line"
    args.dataset = "line"
    args.topology = "erdos_random"
    args.comm = "imperfect"
    args.iid = False
    args.loss_function = parse_loss_function(args.model, args.dataset)

    # setup
    graph = GraphHandler(args.num_users, topology=args.topology)   
    comm = CommModel(graph, args.comm)
    data = DataHandler(args.dataset, args.num_users, args.iid)
    model = ModelHandler(args, data)
    devices = DeviceHandler(args, data, model)
    policy = Policy(graph, policy=args.policy)
    resource = Resource()
    constraint = 30000
    args.T = 200
    # verify setup
    print(args)
    graph.plot()
    data.train.plot()
    data.test.plot()
    labels={0:"S_t=1", 1:"S_t=3", 2:"S_t=6", 3:"S_t=20", 4:"S_t=50", 5:"S_t=10"}
    K_uniform = resource.get(T=args.T, type="delay_uniform", per_round=1, constraint=constraint, a=3, b=0)
    K_1 = resource.get(T=args.T, type="delay_uniform", per_round=1, constraint=constraint, a = 5, b=0)   
    K_2 = resource.get(T=args.T, type="delay_uniform", per_round=1, constraint=constraint, a = 10, b=0)   
    K_3 = resource.get(T=args.T, type="delay_uniform", per_round=1, constraint=constraint, a = 20, b=0)
    K_4 = resource.get(T=args.T, type="delay_uniform", per_round=1, constraint=constraint, a = 50, b=0)      
    plt.plot(K_uniform, label=labels[0])
    plt.plot(K_1, label=labels[1])
    plt.plot(K_2, label=labels[1])
    plt.plot(K_3, label=labels[1])
    plt.plot(K_4, label=labels[1])
    plt.title("Allocation of slots over rounds")
    plt.xlabel("Round")
    plt.ylabel("# slots")
    plt.legend()
    plt.show()

    num_trials = 10

    precision = 0.001
    probs = np.arange(0,1,precision)
    S = [3,5,10,20,50]
    expected_throughput = np.zeros((len(probs),len(S)))
    for i in range(len(probs)):
        if (i%100 == 0):
            print("evaluating expected",i,len(probs))
        for idx, s in enumerate(S):
            expected_throughput[i,idx] = policy.get_expected_throughput(probs[i],s)
    max_S3 = np.max(expected_throughput[:,0])
    p_3 = probs[np.argmax(expected_throughput[:,0])]
    p_5 = probs[np.argmin(np.abs(expected_throughput[0:int(0.4/0.001),1] - max_S3))]
    p_10 = probs[np.argmin(np.abs(expected_throughput[0:int(0.5/0.001),2] - max_S3))]
    p_20 = probs[np.argmin(np.abs(expected_throughput[0:int(0.5/0.001),3] - max_S3))]
    p_50 = probs[np.argmin(np.abs(expected_throughput[0:int(0.5/0.001),4] - max_S3))]
    print(p_3,p_5, p_10, p_20, p_50)
    print(np.max(expected_throughput[:,0]), expected_throughput[np.argmin(np.abs(expected_throughput[:,1] - max_S3)),1])

    times = []   
    losses = []
    energy = []
    # 0
    trials_losses = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, uniform".format(trial))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(3))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_uniform)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # 1
    trials_losses = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, {}".format(trial,labels[1]))
        policy = Policy(graph, policy="random", p=p_5)
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_1)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # 2
    trials_losses = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, {}".format(trial,labels[1]))
        policy = Policy(graph, policy="random", p=p_10)
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_2)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # 3
    trials_losses = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, {}".format(trial,labels[1]))
        policy = Policy(graph, policy="random", p=p_20)
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_3)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # 4
    trials_losses = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, {}".format(trial,labels[1]))
        policy = Policy(graph, policy="random", p=p_50)
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_4)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # optional save
    if 0:
        np.save("experimentdata/E49_1.npy",np.array([losses]))

    # plot
    plt.plot(times[0], losses[0], "-*", label="{}: {}".format(labels[0], policy.get_optimal_prob_throughput_S(3)))
    plt.plot(times[1], losses[1], "-.", label="{}: {}".format(labels[1], p_5))
    plt.plot(times[2], losses[2], "-.", label="{}: {}".format(labels[1], p_10))
    plt.plot(times[3], losses[3], "-.", label="{}: {}".format(labels[1], p_20))
    plt.plot(times[4], losses[4], "-.", label="{}: {}".format(labels[1], p_50))
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    name = "E49 delay uniform"
    plt.title(name)
    plt.grid(True)
    plt.show()

def Fig_4_11_a_E33(save=False):
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(1)
    random.seed(4)
    np.random.seed(1)

    args.nb=20
    args.lr=0.01
    args.num_users = 20
    args.policy = "random"
    args.model = "line"
    args.dataset = "line"
    args.topology = "erdos_random"
    args.comm = "imperfect"
    args.iid = False
    args.loss_function = parse_loss_function(args.model, args.dataset)

    # setup
    graph = GraphHandler(args.num_users, topology=args.topology)   
    comm = CommModel(graph, args.comm)
    data = DataHandler(args.dataset, args.num_users, args.iid)
    model = ModelHandler(args, data)
    devices = DeviceHandler(args, data, model)
    policy = Policy(graph, policy=args.policy)
    resource = Resource()
    constraint = 100000
    args.T = 200
    num_trials = 10

    # verify setup
    print(args)
    graph.plot()
    data.train.plot()
    data.test.plot()
    labels={0:"S_t=1", 1:"S_t=3", 2:"S_t=6", 3:"S_t=20", 4:"S_t=50"}
    K_uniform = resource.get(T=args.T, type="delay_uniform", per_round=1, constraint=constraint, a=1, b=0)
    K_1 = resource.get(T=args.T, type="delay_uniform", per_round=1, constraint=constraint, a = 3, b=0)   
    K_2 = resource.get(T=args.T, type="delay_uniform", per_round=1, constraint=constraint, a = 6, b=0)   
    K_3 = resource.get(T=args.T, type="delay_uniform", per_round=1, constraint=constraint, a = 20, b=0)   
    K_4 = resource.get(T=args.T, type="delay_uniform", per_round=1, constraint=constraint, a = 50, b=0)
    plt.plot(K_uniform, label=labels[0])
    plt.plot(K_1, label=labels[1])
    plt.plot(K_2, label=labels[2])
    plt.plot(K_3, label=labels[3])
    plt.plot(K_4, label=labels[4])
    plt.title("Allocation of slots over rounds")
    plt.xlabel("Round")
    plt.ylabel("# slots")
    plt.legend()
    plt.show()

    times = []   
    losses = []
    energy = []
    # uniform K
    trials_losses = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, uniform".format(trial))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(1))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_uniform)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # 1
    trials_losses = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, {}".format(trial,labels[1]))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(3))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_1)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # 2
    trials_losses = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, {}".format(trial,labels[2]))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(6))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_2)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # 3
    trials_losses = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, {}".format(trial, labels[3]))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(20))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_3)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # 4
    trials_losses = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, {}".format(trial, labels[4]))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(50))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_4)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # optional save
    if 0:
        np.save("experimentdata/E33.npy",np.array([losses]))

    # plotting
    plt.plot(times[0], losses[0], "-*", label="{}: {}".format(labels[0], policy.get_optimal_prob_throughput_S(1)))
    plt.plot(times[1], losses[1], "-.", label="{}: {}".format(labels[1], policy.get_optimal_prob_throughput_S(3)))
    plt.plot(times[2], losses[2], "--", label="{}: {}".format(labels[2], policy.get_optimal_prob_throughput_S(6)))
    plt.plot(times[3], losses[3], "^-", label="{}: {}".format(labels[3], policy.get_optimal_prob_throughput_S(20)))
    plt.plot(times[4], losses[4], ".", label="{}: {}".format(labels[4], policy.get_optimal_prob_throughput_S(50)))
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    name = "E33 delay uniform"
    plt.title(name)
    plt.grid(True)
    plt.show()

def Fig_4_11_c_E34(save=False):
    """NOTE: You must first run E33 and save the results"""

    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(1)
    random.seed(4)
    np.random.seed(1)

    args.nb=20
    args.lr=0.01
    args.num_users = 20
    args.policy = "random"
    args.model = "line"
    args.dataset = "line"
    args.topology = "erdos_random"
    args.comm = "imperfect"
    args.iid = False
    args.loss_function = parse_loss_function(args.model, args.dataset)

    # setup
    graph = GraphHandler(args.num_users, topology=args.topology)   
    args.T = 200

    times = [t for t in range(args.T + 1)]

    losses = np.load("experimentdata/E33.npy")[0]
    
    fig, ax = plt.subplots(1)
    plt.plot([t for t in times], losses[0], "-", color="black", linewidth=1)
    plt.plot([3*t for t in times], losses[1], "-", color="black", linewidth=1)
    plt.plot([6*t for t in times], losses[2], "-", color="black", linewidth=1)
    plt.plot([20*t for t in times], losses[3], "-", color="black", linewidth=1)
    plt.plot([50*t for t in times], losses[4], "-", color="black", linewidth=1)
    
    plt.plot([t for t in times][-1], losses[0][-1], "o", linewidth=1)
    plt.plot([3*t for t in times][-1], losses[1][-1], "o", linewidth=1)
    plt.plot([6*t for t in times][-1], losses[2][-1], "o", linewidth=1)
    plt.plot([20*t for t in times][-1], losses[3][-1], "o", linewidth=1)
    plt.plot([50*t for t in times][-1]-100, losses[4][-1], "o", linewidth=1)

    hfont1= {'fontname':FONTNAME,'fontsize':16,'fontstyle':'normal'}
    hfont2= {'fontname':FONTNAME,'fontsize':18,'fontstyle':'normal'}
    plt.text([t for t in times][-1], losses[0][-1], "S=1", **hfont1)
    plt.text([3*t for t in times][-1], losses[1][-1], "S=3", **hfont1)
    plt.text([6*t for t in times][-1],losses[2][-1], "S=6", **hfont1)
    plt.text([20*t for t in times][-1], losses[3][-1]+0.05, "S=20", **hfont1)
    plt.text([50*t for t in times][-1]-1250, losses[4][-1] + 0.05, "S=50", **hfont1)
    plt.hlines(y=losses[4][-1], xmin=0, xmax=10000, linewidth=0.5, color='purple',linestyles="dashed")

    plt.xlabel("Number of used slots", **hfont2)
    plt.ylabel(r"Loss $\overline{L}$", **hfont2)
    plt.xticks([0,2000,4000,6000,8000,10000], [0,2000,4000,6000,8000,10000], fontsize=13)
    plt.yticks([2.5,3.0,3.5,4.0,4.5,5.0,5.5], [2.5,3.0,3.5,4.0,4.5,5.0,5.5], fontsize=13)
    plt.xlim(0,10000)
    plt.ylim(2.5, 5.5)
    plt.show()

def Fig_4_12_a_E22(save=False):
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(1)
    random.seed(4)
    np.random.seed(1)

    args.nb=20
    args.lr=0.01
    args.num_users = 20
    args.policy = "random"
    args.model = "line"
    args.dataset = "line"
    args.topology = "erdos_random"
    args.comm = "imperfect"
    args.iid = False
    args.loss_function = parse_loss_function(args.model, args.dataset)

    # setup
    graph = GraphHandler(args.num_users, topology=args.topology)   
    comm = CommModel(graph, args.comm)
    data = DataHandler(args.dataset, args.num_users, args.iid)
    model = ModelHandler(args, data)
    devices = DeviceHandler(args, data, model)
    policy = Policy(graph, policy=args.policy)
    resource = Resource()
    constraint = 300
    args.T = 200

    # verify setup
    print(args)
    graph.plot()
    data.train.plot()
    data.test.plot()
    K_uniform = resource.get(T=args.T, type="delay_uniform", per_round=1, constraint=constraint, a=1, b=0)
    labels={0:"S_t=1", 1:"S_t=3", 2:"S_t=6", 3:"S_t=20", 4:"S_t=50", 5:"S_t=10"}
    K_1 = resource.get(T=args.T, type="delay_uniform", per_round=1, constraint=constraint, a = 3, b=0)   
    K_2 = resource.get(T=args.T, type="delay_uniform", per_round=1, constraint=constraint, a = 6, b=0)   
    K_3 = resource.get(T=args.T, type="delay_uniform", per_round=1, constraint=constraint, a = 20, b=0)   
    K_4 = resource.get(T=args.T, type="delay_uniform", per_round=1, constraint=constraint, a = 50, b=0)   
    plt.plot(K_uniform, label=labels[0])
    plt.plot(K_1, label=labels[1])
    plt.plot(K_2, label=labels[2])
    plt.plot(K_3, label=labels[3])
    plt.plot(K_4, label=labels[4])
    plt.title("Allocation of slots over rounds")
    plt.xlabel("Round")
    plt.ylabel("# slots")
    plt.legend()
    plt.show()

    num_trials = 10

    times = []   
    losses = []
    energy = []
    # uniform K
    trials_losses = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, uniform".format(trial))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(1))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_uniform)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # 1
    trials_losses = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, {}".format(trial,labels[1]))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(3))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_1)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # 2
    trials_losses = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, {}".format(trial,labels[2]))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(6))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_2)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # 3
    trials_losses = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, {}".format(trial, labels[3]))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(20))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_3)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # 4
    trials_losses = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, {}".format(trial, labels[4]))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(50))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_4)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # optional save
    if 0:
        np.save("experimentdata/E22.npy",np.array([losses]))
    # plotting
    plt.plot(times[0], losses[0], "-*", label="{}: {}".format(labels[0], policy.get_optimal_prob_throughput_S(1)))
    plt.plot(times[1][0:np.argmin(losses[1])], losses[1][0:np.argmin(losses[1])], "-.", label="{}: {}".format(labels[1], policy.get_optimal_prob_throughput_S(3)))
    plt.plot(times[2][0:np.argmin(losses[2])], losses[2][0:np.argmin(losses[2])], "--", label="{}: {}".format(labels[2], policy.get_optimal_prob_throughput_S(6)))
    plt.plot(times[3][0:np.argmin(losses[3])], losses[3][0:np.argmin(losses[3])], "^-", label="{}: {}".format(labels[3], policy.get_optimal_prob_throughput_S(20)))
    plt.plot(times[4][0:np.argmin(losses[4])], losses[4][0:np.argmin(losses[4])], ".", label="{}: {}".format(labels[4], policy.get_optimal_prob_throughput_S(50)))
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    name = "E22 delay uniform"
    plt.title(name)
    plt.grid(True)
    plt.show()

def Fig_4_12_b_E38(save=False):
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(1)
    random.seed(4)
    np.random.seed(1)

    args.status_freq = 100
    args.nb=20
    args.lr=0.01
    args.num_users = 20
    args.policy = "random"
    args.model = "linear_model"
    args.dataset = "4clusters"
    args.topology = "erdos_random"
    args.comm = "imperfect"
    args.iid = False
    args.loss_function = parse_loss_function(args.model, args.dataset)

    # setup
    graph = GraphHandler(args.num_users, topology=args.topology)   
    comm = CommModel(graph, args.comm)
    data = DataHandler(args.dataset, args.num_users, args.iid)
    model = ModelHandler(args, data)
    devices = DeviceHandler(args, data, model)
    policy = Policy(graph, policy=args.policy)
    resource = Resource()
    constraint = 5000
    args.T = 5001

    # verify setup
    print(args)
    graph.plot()
    data.train.plot()
    data.test.plot()
    K_uniform = resource.get(T=args.T, type="delay_uniform", per_round=1, constraint=constraint, a=1, b=0)
    labels={0:"S_t=1", 1:"S_t=3", 2:"S_t=6", 3:"S_t=20", 4:"S_t=50", 5:"S_t=10"}
    K_1 = resource.get(T=args.T, type="delay_uniform", per_round=1, constraint=constraint, a = 3, b=0)   
    K_2 = resource.get(T=args.T, type="delay_uniform", per_round=1, constraint=constraint, a = 6, b=0)   
    K_3 = resource.get(T=args.T, type="delay_uniform", per_round=1, constraint=constraint, a = 20, b=0)   
    K_4 = resource.get(T=args.T, type="delay_uniform", per_round=1, constraint=constraint, a = 50, b=0)   
    plt.plot(K_uniform, label=labels[0])
    plt.plot(K_1, label=labels[1])
    plt.plot(K_2, label=labels[2])
    plt.plot(K_3, label=labels[3])
    plt.plot(K_4, label=labels[4])
    plt.title("Allocation of slots over rounds")
    plt.xlabel("Round")
    plt.ylabel("# slots")
    plt.legend()
    plt.show()

    num_trials = 10

    times = []   
    losses = []
    accuracies = []
    energy = []
    # uniform K
    trials_losses = []
    trials_accuracies = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, uniform".format(trial))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(1))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_uniform)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_accuracies.append([p[1][0] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    accuracies.append(list(np.mean(np.array(trials_accuracies),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # 1
    trials_losses = []
    trials_accuracies = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, {}".format(trial,labels[1]))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(3))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_1)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_accuracies.append([p[1][0] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    accuracies.append(list(np.mean(np.array(trials_accuracies),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # 2
    trials_losses = []
    trials_accuracies = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, {}".format(trial,labels[2]))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(6))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_2)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_accuracies.append([p[1][0] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    accuracies.append(list(np.mean(np.array(trials_accuracies),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # 3
    trials_losses = []
    trials_accuracies = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, {}".format(trial, labels[3]))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(20))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_3)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_accuracies.append([p[1][0] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    accuracies.append(list(np.mean(np.array(trials_accuracies),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # 4
    trials_losses = []
    trials_accuracies = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, {}".format(trial, labels[4]))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(50))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_4)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_accuracies.append([p[1][0] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    accuracies.append(list(np.mean(np.array(trials_accuracies),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # optional save
    if 0:
        np.save("experimentdata/E38.npy",np.array([accuracies, losses]))
    # plotting
    plt.plot(times[0], losses[0], "-*", label="{}: {}".format(labels[0], policy.get_optimal_prob_throughput_S(1)))
    plt.plot(times[1][0:np.argmin(losses[1])], losses[1][0:np.argmin(losses[1])], "-.", label="{}: {}".format(labels[1], policy.get_optimal_prob_throughput_S(3)))
    plt.plot(times[2][0:np.argmin(losses[2])], losses[2][0:np.argmin(losses[2])], "--", label="{}: {}".format(labels[2], policy.get_optimal_prob_throughput_S(6)))
    plt.plot(times[3][0:np.argmin(losses[3])], losses[3][0:np.argmin(losses[3])], "^-", label="{}: {}".format(labels[3], policy.get_optimal_prob_throughput_S(20)))
    plt.plot(times[4][0:np.argmin(losses[4])], losses[4][0:np.argmin(losses[4])], ".", label="{}: {}".format(labels[4], policy.get_optimal_prob_throughput_S(50)))
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    name = "E38 delay uniform"
    plt.title(name)
    plt.grid(True)
    plt.show()

def Fig_4_13_a_E22(save=False):
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(1)
    random.seed(4)
    np.random.seed(1)

    args.nb=20
    args.lr=0.01
    args.num_users = 20
    args.policy = "random"
    args.model = "line"
    args.dataset = "line"
    args.topology = "erdos_random"
    args.comm = "imperfect"
    args.iid = False
    args.loss_function = parse_loss_function(args.model, args.dataset)

    # setup
    graph = GraphHandler(args.num_users, topology=args.topology)   
    comm = CommModel(graph, args.comm)
    data = DataHandler(args.dataset, args.num_users, args.iid)
    model = ModelHandler(args, data)
    devices = DeviceHandler(args, data, model)
    policy = Policy(graph, policy=args.policy)
    resource = Resource()
    constraint = 300
    args.T = 200

    # verify setup
    print(args)
    graph.plot()
    data.train.plot()
    data.test.plot()
    K_uniform = resource.get(T=args.T, type="delay_uniform", per_round=1, constraint=constraint, a=1, b=0)
    labels={0:"S_t=1", 1:"S_t=3", 2:"S_t=6", 3:"S_t=20", 4:"S_t=50", 5:"S_t=10"}
    K_1 = resource.get(T=args.T, type="delay_uniform", per_round=1, constraint=constraint, a = 3, b=0)   
    K_2 = resource.get(T=args.T, type="delay_uniform", per_round=1, constraint=constraint, a = 6, b=0)   
    K_3 = resource.get(T=args.T, type="delay_uniform", per_round=1, constraint=constraint, a = 20, b=0)   
    K_4 = resource.get(T=args.T, type="delay_uniform", per_round=1, constraint=constraint, a = 50, b=0)   
    plt.plot(K_uniform, label=labels[0])
    plt.plot(K_1, label=labels[1])
    plt.plot(K_2, label=labels[2])
    plt.plot(K_3, label=labels[3])
    plt.plot(K_4, label=labels[4])
    plt.title("Allocation of slots over rounds")
    plt.xlabel("Round")
    plt.ylabel("# slots")
    plt.legend()
    plt.show()

    num_trials = 10

    times = []   
    losses = []
    energy = []
    # uniform K
    trials_losses = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, uniform".format(trial))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(1))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_uniform)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # 1
    trials_losses = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, {}".format(trial,labels[1]))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(3))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_1)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # 2
    trials_losses = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, {}".format(trial,labels[2]))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(6))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_2)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # 3
    trials_losses = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, {}".format(trial, labels[3]))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(20))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_3)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # 4
    trials_losses = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, {}".format(trial, labels[4]))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(50))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_4)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # optional save
    if 0:
        np.save("experimentdata/E22.npy",np.array([losses]))
    # plotting
    plt.plot(times[0], losses[0], "-*", label="{}: {}".format(labels[0], policy.get_optimal_prob_throughput_S(1)))
    plt.plot(times[1][0:np.argmin(losses[1])], losses[1][0:np.argmin(losses[1])], "-.", label="{}: {}".format(labels[1], policy.get_optimal_prob_throughput_S(3)))
    plt.plot(times[2][0:np.argmin(losses[2])], losses[2][0:np.argmin(losses[2])], "--", label="{}: {}".format(labels[2], policy.get_optimal_prob_throughput_S(6)))
    plt.plot(times[3][0:np.argmin(losses[3])], losses[3][0:np.argmin(losses[3])], "^-", label="{}: {}".format(labels[3], policy.get_optimal_prob_throughput_S(20)))
    plt.plot(times[4][0:np.argmin(losses[4])], losses[4][0:np.argmin(losses[4])], ".", label="{}: {}".format(labels[4], policy.get_optimal_prob_throughput_S(50)))
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    name = "E22 delay uniform"
    plt.title(name)
    plt.grid(True)
    plt.show()

def Fig_4_13_b_E38(save=False):
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(1)
    random.seed(4)
    np.random.seed(1)

    args.status_freq = 100
    args.nb=20
    args.lr=0.01
    args.num_users = 20
    args.policy = "random"
    args.model = "linear_model"
    args.dataset = "4clusters"
    args.topology = "erdos_random"
    args.comm = "imperfect"
    args.iid = False
    args.loss_function = parse_loss_function(args.model, args.dataset)

    # setup
    graph = GraphHandler(args.num_users, topology=args.topology)   
    comm = CommModel(graph, args.comm)
    data = DataHandler(args.dataset, args.num_users, args.iid)
    model = ModelHandler(args, data)
    devices = DeviceHandler(args, data, model)
    policy = Policy(graph, policy=args.policy)
    resource = Resource()
    constraint = 5000
    args.T = 5001

    # verify setup
    print(args)
    graph.plot()
    data.train.plot()
    data.test.plot()
    K_uniform = resource.get(T=args.T, type="delay_uniform", per_round=1, constraint=constraint, a=1, b=0)
    labels={0:"S_t=1", 1:"S_t=3", 2:"S_t=6", 3:"S_t=20", 4:"S_t=50", 5:"S_t=10"}
    K_1 = resource.get(T=args.T, type="delay_uniform", per_round=1, constraint=constraint, a = 3, b=0)   
    K_2 = resource.get(T=args.T, type="delay_uniform", per_round=1, constraint=constraint, a = 6, b=0)   
    K_3 = resource.get(T=args.T, type="delay_uniform", per_round=1, constraint=constraint, a = 20, b=0)   
    K_4 = resource.get(T=args.T, type="delay_uniform", per_round=1, constraint=constraint, a = 50, b=0)   
    plt.plot(K_uniform, label=labels[0])
    plt.plot(K_1, label=labels[1])
    plt.plot(K_2, label=labels[2])
    plt.plot(K_3, label=labels[3])
    plt.plot(K_4, label=labels[4])
    plt.title("Allocation of slots over rounds")
    plt.xlabel("Round")
    plt.ylabel("# slots")
    plt.legend()
    plt.show()

    num_trials = 10

    times = []   
    losses = []
    accuracies = []
    energy = []
    # uniform K
    trials_losses = []
    trials_accuracies = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, uniform".format(trial))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(1))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_uniform)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_accuracies.append([p[1][0] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    accuracies.append(list(np.mean(np.array(trials_accuracies),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # 1
    trials_losses = []
    trials_accuracies = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, {}".format(trial,labels[1]))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(3))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_1)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_accuracies.append([p[1][0] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    accuracies.append(list(np.mean(np.array(trials_accuracies),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # 2
    trials_losses = []
    trials_accuracies = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, {}".format(trial,labels[2]))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(6))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_2)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_accuracies.append([p[1][0] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    accuracies.append(list(np.mean(np.array(trials_accuracies),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # 3
    trials_losses = []
    trials_accuracies = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, {}".format(trial, labels[3]))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(20))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_3)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_accuracies.append([p[1][0] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    accuracies.append(list(np.mean(np.array(trials_accuracies),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # 4
    trials_losses = []
    trials_accuracies = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, {}".format(trial, labels[4]))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(50))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_4)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_accuracies.append([p[1][0] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    accuracies.append(list(np.mean(np.array(trials_accuracies),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # optional save
    if 0:
        np.save("experimentdata/E38.npy",np.array([accuracies, losses]))
    # plotting
    plt.plot(times[0], losses[0], "-*", label="{}: {}".format(labels[0], policy.get_optimal_prob_throughput_S(1)))
    plt.plot(times[1][0:np.argmin(losses[1])], losses[1][0:np.argmin(losses[1])], "-.", label="{}: {}".format(labels[1], policy.get_optimal_prob_throughput_S(3)))
    plt.plot(times[2][0:np.argmin(losses[2])], losses[2][0:np.argmin(losses[2])], "--", label="{}: {}".format(labels[2], policy.get_optimal_prob_throughput_S(6)))
    plt.plot(times[3][0:np.argmin(losses[3])], losses[3][0:np.argmin(losses[3])], "^-", label="{}: {}".format(labels[3], policy.get_optimal_prob_throughput_S(20)))
    plt.plot(times[4][0:np.argmin(losses[4])], losses[4][0:np.argmin(losses[4])], ".", label="{}: {}".format(labels[4], policy.get_optimal_prob_throughput_S(50)))
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    name = "E38 delay uniform"
    plt.title(name)
    plt.grid(True)
    plt.show()

def Fig_4_14_a_E52_2(save=False):
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(1)
    random.seed(4)
    np.random.seed(1)

    args.T = 301
    args.nb=20
    args.lr=0.01
    args.num_users = 20
    args.policy = "random"
    args.model = "line"
    args.dataset = "line"
    args.topology = "erdos_random"
    args.comm = "imperfect"
    args.iid = False
    args.loss_function = parse_loss_function(args.model, args.dataset)

    # setup
    graph = GraphHandler(args.num_users, topology=args.topology)   
    comm = CommModel(graph, args.comm)
    data = DataHandler(args.dataset, args.num_users, args.iid)
    model = ModelHandler(args, data)
    devices = DeviceHandler(args, data, model)
    policy = Policy(graph, policy=args.policy)
    resource = Resource()
    constraint = 300

    # verify setup
    print(args)
    graph.plot()
    data.train.plot()
    data.test.plot()
    labels={0:"b=1", 1:"b=2", 2:"b=3", 3:"b=4", 4:"b=5", 5:"S_t=10"}
    K_uniform = resource.get(T=args.T, type="every_nth", constraint=constraint, a=1, b=2)
    K_1 = resource.get(T=args.T, type="every_nth", constraint=constraint, a = 3, b=2)   
    K_2 = resource.get(T=args.T, type="every_nth", constraint=constraint, a = 6, b=2)   
    K_3 = resource.get(T=args.T, type="every_nth", constraint=constraint, a = 20, b=2)   
    K_4 = resource.get(T=args.T, type="every_nth", constraint=constraint, a = 50, b=2)   

    num_trials = 10

    times = []   
    losses = []
    energy = []
    # uniform K
    trials_losses = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, uniform".format(trial))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(1))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_uniform, auto_prob=True)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # 1
    trials_losses = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, {}".format(trial,labels[1]))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(3))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_1, auto_prob=True)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # 2
    trials_losses = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, {}".format(trial,labels[2]))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(6))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_2, auto_prob=True)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # 3
    trials_losses = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, {}".format(trial, labels[3]))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(20))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_3, auto_prob=True)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # 4
    trials_losses = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, {}".format(trial, labels[4]))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(50))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_4, auto_prob=True)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # optional save
    if 0:
        np.save("experimentdata/E52_2.npy",np.array([losses]))
    # plotting
    plt.plot(times[0], losses[0], "-*", label="{}: {}".format(labels[0], policy.get_optimal_prob_throughput_S(1)))
    plt.plot(times[1], losses[1], "-.", label="{}, S={}".format(labels[1], 6))
    plt.plot(times[2], losses[2], "--", label="{}, S={}".format(labels[2], 6))
    plt.plot(times[3], losses[3], "^-", label="{}, S={}".format(labels[3], 6))
    plt.plot(times[4], losses[4], ".", label="{}, S={}".format(labels[4], 6))
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    name = "E48 every nth"
    plt.title(name)
    plt.grid(True)
    plt.show()

def Fig_4_14_b_E53_2(save=False):
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(1)
    random.seed(4)
    np.random.seed(1)

    args.status_freq = 100
    args.T = 5001
    args.nb=20
    args.lr=0.01
    args.num_users = 20
    args.policy = "random"
    args.model = "linear_model"
    args.dataset = "4clusters"
    args.topology = "erdos_random"
    args.comm = "imperfect"
    args.iid = False
    args.loss_function = parse_loss_function(args.model, args.dataset)

    # setup
    graph = GraphHandler(args.num_users, topology=args.topology)   
    comm = CommModel(graph, args.comm)
    data = DataHandler(args.dataset, args.num_users, args.iid)
    model = ModelHandler(args, data)
    devices = DeviceHandler(args, data, model)
    policy = Policy(graph, policy=args.policy)
    resource = Resource()
    constraint = 5000

    # verify setup
    print(args)
    graph.plot()
    data.train.plot()
    data.test.plot()
    labels={0:"b=1", 1:"b=2", 2:"b=3", 3:"b=4", 4:"b=5", 5:"S_t=10"}
    K_uniform = resource.get(T=args.T, type="every_nth", constraint=constraint, a=1, b=2)
    K_1 = resource.get(T=args.T, type="every_nth", constraint=constraint, a = 3, b=2)   
    K_2 = resource.get(T=args.T, type="every_nth", constraint=constraint, a = 6, b=2)   
    K_3 = resource.get(T=args.T, type="every_nth", constraint=constraint, a = 20, b=2)   
    K_4 = resource.get(T=args.T, type="every_nth", constraint=constraint, a = 50, b=2)   

    num_trials = 10

    times = []   
    losses = []
    accuracies = []
    energy = []
    # uniform K
    trials_losses = []
    trials_accuracies = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, uniform".format(trial))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(1))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_uniform, auto_prob=True)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_accuracies.append([p[1][0] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    accuracies.append(list(np.mean(np.array(trials_accuracies),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # 1
    trials_losses = []
    trials_accuracies = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, {}".format(trial,labels[1]))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(3))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_1, auto_prob=True)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_accuracies.append([p[1][0] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    accuracies.append(list(np.mean(np.array(trials_accuracies),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # 2
    trials_losses = []
    trials_accuracies = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, {}".format(trial,labels[2]))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(6))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_2, auto_prob=True)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_accuracies.append([p[1][0] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    accuracies.append(list(np.mean(np.array(trials_accuracies),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # 3
    trials_losses = []
    trials_accuracies = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, {}".format(trial, labels[3]))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(12))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_3, auto_prob=True)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_accuracies.append([p[1][0] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    accuracies.append(list(np.mean(np.array(trials_accuracies),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # 4
    trials_losses = []
    trials_accuracies = []
    trials_energies = []
    for trial in range(num_trials):
        print("Trial #{}, {}".format(trial, labels[4]))
        policy = Policy(graph, policy="random", p=policy.get_optimal_prob_throughput_S(24))
        result = dfl(graph.W, data.test, devices.devices, policy, comm, args, S=K_4, auto_prob=True)
        performance = result["performance"]
        energies = result["energies"]
        times.append([p[0] for p in performance])
        trials_losses.append([p[1][-1] for p in performance])
        trials_accuracies.append([p[1][0] for p in performance])
        trials_energies.append([e[1] for e in energies])
        times.append([p[0] for p in performance])
    losses.append(list(np.mean(np.array(trials_losses),axis=0)))
    accuracies.append(list(np.mean(np.array(trials_accuracies),axis=0)))
    energy.append(list(np.mean(np.array(trials_energies),axis=0)))

    # optional save
    if 0:
        np.save("experimentdata/E53_2.npy",np.array([losses, accuracies]))
    # plotting
    plt.plot(times[0], losses[0], "-*", label="{}: {}".format(labels[0], policy.get_optimal_prob_throughput_S(1)))
    plt.plot(times[1], losses[1], "-.", label="{}, S={}".format(labels[1], 6))
    plt.plot(times[2], losses[2], "--", label="{}, S={}".format(labels[2], 6))
    plt.plot(times[3], losses[3], "^-", label="{}, S={}".format(labels[3], 6))
    plt.plot(times[4], losses[4], ".", label="{}, S={}".format(labels[4], 6))
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    name = "E48 every nth"
    plt.title(name)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    Fig_3_1()
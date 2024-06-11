import argparse
import numpy as np
import torch
import learner as ln
from data import SPData
from PS_method import PSmethod
import os
from time import perf_counter
from mznn import MZNN
from visualize import plot_heat, plot_anime, plot_cost_constraint
import logging
import datetime

def update_lag_mul(data, net):
    net.update_lag_mul(data.X_train["interval"], data.X_train["bd"], data.y_train["bd"])
    # output mean cost over several trajectories
    cost, hmin = compute_cost_hmin(data.X_train["interval"], net, net.trajs)
    return cost, hmin


def compute_cost_hmin(t, net, traj_count):
    cost = torch.mean(net.value_function(t)[:traj_count]).item()
    hmin = net.hmin_function(t, traj_count).item()
    logging.info("cost value: {}\t".format(cost))
    logging.info("constraint value: {}\n".format(hmin))
    return cost, hmin


def main():
    parser = argparse.ArgumentParser(description="Hyperparameters to be tuned")
    parser.add_argument("--l", type=float, default=0.001, help="l in the soft penalty")
    parser.add_argument(
        "--eps", type=float, default=0.00015, help="epsilon in the soft penalty"
    )
    parser.add_argument(
        "--lam", type=float, default=600.0, help="weight of boundary loss"
    )
    parser.add_argument("--ntype", type=str, default="G", help="type of NN")
    parser.add_argument("--layers", type=int, default=6, help="layers of NN")
    parser.add_argument("--width", type=int, default=60, help="width of NN")
    parser.add_argument("--act", type=str, default="relu", help="activation function")
    parser.add_argument(
        "--lagmulfreq", type=int, default=1, help="the frequency of updating lagmul"
    )
    parser.add_argument("--rho", type=float, default=1.0, help="parameter for aug Lag")
    parser.add_argument(
        "--iters", type=int, default=100000, help="number of iterations"
    )
    parser.add_argument(
        "--lbfgsiters",
        type=int,
        default=100,
        help="number of lbfgs iterations for testcase2",
    )
    # modify the following flags to test
    parser.add_argument(
        "--testcase", type=int, default=2, help="1 for comparison, 2 for mul traj"
    )
    parser.add_argument(
        "--penalty", action="store_true", help="whether use penalty function"
    )
    parser.add_argument(
        "--addloss",
        type=int,
        default=0,
        help="0 for no added loss, 1 for log penalty or aug Lag, 2 for quadratic penalty",
    )
    parser.add_argument("--adddim", type=int, default=0, help="added dimension")
    parser.add_argument("--gpu", action="store_true", help="whether to use gpu")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    # use this flag if loading instead of training
    parser.add_argument(
        "--loadno", type=int, default=0, help="the no. for the model to load"
    )
    parser.add_argument(
        "--loadfoldername", type=str, default="", help="the foldername for the model to load"
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # set up logging

    if args.gpu:
        device = "gpu"  # 'cpu' or 'gpu'
    else:
        device = "cpu"  # 'cpu' or 'gpu'

    # data
    t_terminal = 1.0
    train_num = 200
    test_num = 200
    train_traj = 1
    test_traj = 1
    train_noise = 0.0
    test_noise = 0.0
    traj_count = 1
    # change to mult traj if testcase == 2
    if args.testcase == 2:
        train_traj = 100
        test_traj = 100
        train_noise = 1.0
        num_interpolate = 3
        traj_count = num_interpolate * 2

    qr = 0.5  # width of the obstacle
    dr = 0.5  # radius of drone
    ql = [1.1, 1.1]  # length of the obstacle
    ws = [[-2.5, 0], [1.4, 0]]  # starting points of the obstacle
    angles = [0, 0]  # angles of the obstacle
    q_initial = [-2, -2, 2, -2, 2, 2, -2, 2]
    q_terminal = [2, 2, -2, 2, -2, -2, 2, -2]

    dim = len(q_initial)
    add_dim = args.adddim
    rho = args.rho
    l = args.l
    eps = args.eps
    lam = args.lam
    ntype = args.ntype
    layers = args.layers
    width = args.width
    activation = args.act
    dtype = "double"

    # training
    lr = 0.001
    iterations = args.iters
    print_every = 1000
    batch_size = None

    # TODO: figure out a better output folder name
    foldername = "testcase_{}_distsquare_".format(args.testcase)
    foldername += "eps_{}_l_{}_".format(eps, l)
    if args.penalty:
        foldername += "penalty_"
    else:
        foldername += "auglag_"
    foldername += "lbfgsiters_{}_".format(args.lbfgsiters)
    foldername += "addloss_{}_".format(args.addloss)
    foldername += "adddim_{}_".format(add_dim)
    foldername += "seed_{}".format(args.seed)
    figname = foldername
    
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    if not os.path.exists("logs"):
        os.mkdir("logs")
    if not os.path.exists("logs/" + foldername):
        os.mkdir("logs/" + foldername)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(module)s.%(funcName)s: Line%(lineno)d] - %(levelname)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        filename=f"logs/{foldername}/{now_str}.log",
        filemode="w",
    )
    
    if not os.path.exists("figs"):
        os.mkdir("figs")
    if not os.path.exists("figs/" + figname):
        os.mkdir("figs/" + figname)

    data = SPData(train_num, test_num, train_traj, test_traj, train_noise, test_noise, q_terminal, t_terminal, q_initial)
    if args.loadno == 0:
        net = MZNN(dim, layers, width, activation, ntype, dr=dr, ws=ws, angles=angles, qr=qr, ql=ql, l=l, eps=eps, lam=lam, add_dim=add_dim, ifpenalty=args.penalty, rho=rho, add_loss=args.addloss, update_lagmul_freq=args.lagmulfreq, trajs=train_traj, dtype=dtype, device=device)
        callback = update_lag_mul

        args_nn = {
            "data": data,
            "net": net,
            "criterion": None,
            "optimizer": "adam",
            "lr": lr,
            "iterations": iterations,
            "lbfgs_steps": 0,
            "path": foldername,
            "batch_size": batch_size,
            "print_every": print_every,
            "save": True,
            "callback": callback,
            "dtype": dtype,
            "device": device,
        }

        ln.Brain.Init(**args_nn)
        ln.Brain.Run()
        ln.Brain.Restore()
        ln.Brain.Output()
    else:
        if args.loadfoldername == "":
            raise ValueError("Please provide the foldername for the model to load")
        net = torch.load(
            "model/" + args.foldername + "/model{}.pkl".format(args.loadno),
            map_location=torch.device("cpu"),
        )

    net_plot = net

    X_train, y_train, X_test, y_test = (
        data.X_train,
        data.y_train,
        data.X_test,
        data.y_test,
    )
    if args.testcase == 2:  # multi traj
        y_test['bd'] = y_test['bd'] + torch.tensor(np.concatenate([(2 * np.random.rand(data.train_traj,1,net.dim) - 1), np.zeros((data.train_traj,1,net.dim))], axis = 1), device = net.device, dtype = net.dtype)
        y_test["bd"][num_interpolate, 0] = torch.tensor(
            [-2, -4, 2, -4, 2, 4, -2, 4], device=net.device, dtype=net.dtype
        )
        y_test["bd"][num_interpolate + 1, 0] = torch.tensor(
            [-4, -4, 0, -4, 0, 4, -4, 4], device=net.device, dtype=net.dtype
        )
        y_test["bd"][num_interpolate + 2, 0] = torch.tensor(
            [0, -4, 0, -2, 0, 2, 0, 4], device=net.device, dtype=net.dtype
        )
        # y_test['bd'] = y_test['bd'].float()
        # LBFGS training
        net.LBFGS_training(X_test, y_test, True, args.lbfgsiters)
        q_pred = net.predict_q(X_test["interval"], True)
        plot_heat(
            q_pred,
            net_plot,
            figname + "/NN",
            num_interpolate,
            traj_count,
            y_train,
            y_test,
        )
    else:
        q_pred = net_plot.predict_q(data.X_test["interval"], True)[0, ...]
        plot_anime(q_pred, net_plot, figname + "/NN")
    loss = np.loadtxt("outputs/" + foldername + "/loss.txt")
    plot_cost_constraint(data, net_plot, loss, figname, print_every)

    # print cost and hmin
    logging.info("test cost and hmin:\n")
    compute_cost_hmin(
        data.X_test["interval"], net, traj_count
    )  # only print values for the first traj_count many trajs

    # PS method fine tuning
    # start = perf_counter()
    # # compute the output trajectory of NN
    # num_times = 20
    # num_nodes = 3
    # PSiters = 10000
    # time_endpts = np.linspace(0.0, t_terminal, num_times)
    # q_ps = np.zeros((traj_count, (num_times-1) * num_nodes, dim))
    # # plot initial x
    # #plot_anime(PSsolver.get_initial_x(), net, figname + '/PSinit')
    # for i in range(traj_count):
    #     # set initialization for PS method
    #     x_init_ps = y_test['bd'][i,0].detach().cpu().numpy()
    #     x_term_ps = y_test['bd'][i,1].detach().cpu().numpy()
    #     PSsolver = PSmethod(time_endpts, num_nodes, net.dim, net, x_init_ps, x_term_ps, i)
    #     PSsolver.solve(PSiters)
    #     q_ps[i] = PSsolver.get_x()
    #     logging.info(' \n')
    # end = perf_counter()
    # execution_time = (end - start)
    # logging.info('PS running time: {}'.format(execution_time))
    # TODO: this q_ps is not time-uniformly distributed. How to generate anime for this?
    # if args.testcase == 2:
    #     plot_heat(q_ps, net, figname+'/PSmethod', num_interpolate, traj_count, y_train, y_test)
    # else:
    #     plot_anime(q_ps[0], net, figname+'/PSmethod')


if __name__ == "__main__":
    main()

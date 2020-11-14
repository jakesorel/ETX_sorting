import numpy as np
# try:
from self_organisation import *
# except ModuleNotFoundError:
#     from lattice_model.self_organisation import *
import dask
from dask.distributed import Client
import os
import sys
from scipy.sparse import csc_matrix,save_npz
from itertools import permutations



def do_jobWEE(sig):
    N_E, N_T, N_X, scaler = 6, 6, 8, 3
    division = False
    seed_type = "random_circle"
    medium_energy = 5e2
    num_x, num_y = 25,25
    T_t = np.logspace(3,-1,5*10**4)
    W_EE,W_ET,W_TT,W_EX,W_TX,W_XX = 8,4,8,2,2,0
    W0 = np.array([[W_EE, W_ET, W_EX],
                   [W_ET, W_TT, W_TX],
                   [W_EX, W_TX, W_XX]])
    sigma0 = np.array([[0,0,0,0],[0,sig,0,0],[0,0,0,0],[0,0,0,0]])
    etx = ETX()
    etx.generate_lattice(num_x=num_x, num_y=num_y)
    etx.boundary_definition()
    etx.define_running_params(T_t=T_t)
    etx.make_C0(N_E, N_T, N_X, scaler, seed_type=seed_type)
    etx.make_ID0(division=division)
    etx.define_saving_params()
    etx.define_interaction_energies(W0, medium_energy,sigma0=sigma0)
    etx.initialise_simulation()
    etx.perform_simulation(end_only=True)
    etx.get_C_save()
    C_end = etx.C_save[-1]
    return C_end.astype(np.int64)


def do_jobWET(sig):
    N_E, N_T, N_X, scaler = 6, 6, 8, 3
    division = False
    seed_type = "random_circle"
    medium_energy = 5e2
    num_x, num_y = 25,25
    T_t = np.logspace(3,-1,5*10**4)
    W_EE,W_ET,W_TT,W_EX,W_TX,W_XX = 8,4,8,2,2,0
    W0 = np.array([[W_EE, W_ET, W_EX],
                   [W_ET, W_TT, W_TX],
                   [W_EX, W_TX, W_XX]])
    sigma0 = np.array([[0,0,0,0],[0,0,sig,0],[0,sig,0,0],[0,0,0,0]])
    etx = ETX()
    etx.generate_lattice(num_x=num_x, num_y=num_y)
    etx.boundary_definition()
    etx.define_running_params(T_t=T_t)
    etx.make_C0(N_E, N_T, N_X, scaler, seed_type=seed_type)
    etx.make_ID0(division=division)
    etx.define_saving_params()
    etx.define_interaction_energies(W0, medium_energy,sigma0=sigma0)
    etx.initialise_simulation()
    etx.perform_simulation(end_only=True)
    etx.get_C_save()
    C_end = etx.C_save[-1]
    return C_end.astype(np.int64)



def do_jobWTT(sig):
    N_E, N_T, N_X, scaler = 6, 6, 8, 3
    division = False
    seed_type = "random_circle"
    medium_energy = 5e2
    num_x, num_y = 25,25
    T_t = np.logspace(3,-1,5*10**4)
    W_EE,W_ET,W_TT,W_EX,W_TX,W_XX = 8,4,8,2,2,0
    W0 = np.array([[W_EE, W_ET, W_EX],
                   [W_ET, W_TT, W_TX],
                   [W_EX, W_TX, W_XX]])
    sigma0 = np.array([[0,0,0,0],[0,0,0,0],[0,0,sig,0],[0,0,0,0]])
    etx = ETX()
    etx.generate_lattice(num_x=num_x, num_y=num_y)
    etx.boundary_definition()
    etx.define_running_params(T_t=T_t)
    etx.make_C0(N_E, N_T, N_X, scaler, seed_type=seed_type)
    etx.make_ID0(division=division)
    etx.define_saving_params()
    etx.define_interaction_energies(W0, medium_energy,sigma0=sigma0)
    etx.initialise_simulation()
    etx.perform_simulation(end_only=True)
    etx.get_C_save()
    C_end = etx.C_save[-1]
    return C_end.astype(np.int64)


if __name__ == "__main__":
    num_x, num_y = 25,25

    sigma_space = np.logspace(0,2.5,20)
    rep_space = np.arange(20)
    SS, RR = np.meshgrid(sigma_space,rep_space,indexing="ij")
    inputs = np.array([SS.ravel(),RR.ravel()]).T
    n_slurm_tasks = int(os.environ["SLURM_NTASKS"])
    client = Client(threads_per_worker=1, n_workers=n_slurm_tasks,memory_limit="1GB")
    lazy_results = []
    for input in inputs:
        sigma = input[0]
        lazy_result = dask.delayed(do_jobWTT)(sigma)
        lazy_results.append(lazy_result)
    results = dask.compute(*lazy_results)


    if not os.path.exists("het_results_WTT"):
        os.makedirs("het_results_WTT")
    np.savetxt("het_results_WTT/params_%d.txt"%int(sys.argv[1]),inputs)
    for i, result in enumerate(results):
        save_npz("het_results_WTT/%d_%d.npz"%(i,int(sys.argv[1])),csc_matrix(result))
    #
    # etx = ETX()
    #
    # subpops = np.zeros((len(results),3))
    # for i, result in enumerate(results):
    #     subpops[i] = etx.find_subpopulations(result)
    #
    # subpops = subpops.reshape((SS.shape[0],SS.shape[1],3))
    #
    # fig, ax = plt.subplots()
    # ax.plot(2/subpops[:, :, :2].sum(axis=-1).mean(axis=-1))
    # fig.show()
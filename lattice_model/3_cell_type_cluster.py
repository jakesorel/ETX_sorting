import numpy as np
# try:
from lattice_model.self_organisation import *
# except ModuleNotFoundError:
#     from lattice_model.self_organisation import *
import dask
from dask.distributed import Client
import os
import sys
from scipy.sparse import csc_matrix,save_npz
from itertools import permutations

four_states = np.array([[0.25,1.75],[1.75,1.25],[0.25,0.75],[1.75,0.25]])

WXX = four_states[0,0]*four_states[1,0]


inputs = np.array([[80,40,80,20,20,0],
                   [100,40,80,20,0,20],
                   [100,40,80,0,20,20],
                   [100,40,80,0,0,20]])

#do_job([60,40,60,20,20,0]) ETX params

do_job([80,60,40,0,20,0])


def do_job(inputt):
    N_E, N_T, N_X, scaler = 6, 6, 6, 3
    division = False
    seed_type = "random_circle"
    medium_energy = 5e2
    num_x, num_y = 25,25
    T_t = np.logspace(3,-1,5*10**4)
    W_EE,W_ET,W_TT,W_EX,W_TX,W_XX = inputt
    W0 = np.array([[W_EE, W_ET, W_EX],
                   [W_ET, W_TT, W_TX],
                   [W_EX, W_TX, W_XX]])
    etx = ETX()
    etx.generate_lattice(num_x=num_x, num_y=num_y)
    etx.boundary_definition()
    etx.define_running_params(T_t=T_t)
    etx.make_C0(N_E, N_T, N_X, scaler, seed_type=seed_type)
    etx.make_ID0(division=division)
    etx.define_saving_params()
    etx.define_interaction_energies(W0, medium_energy)
    etx.initialise_simulation()
    etx.perform_simulation(end_only=True)
    etx.get_C_save()
    C_end = etx.C_save[-1]
    etx.plot_save(C_end,xlim=(0,50),ylim=(0,50))
    return C_end.astype(np.int64)

"""Full conformations"""

#1. three clusters

do_job([2,0,2,0,0,2])

#2. Rings

do_job([8,6,4,0,2,0]) #ETX

do_job([4,6,8,2,0,0]) #TEX

do_job([4,2,0,6,0,8]) #XET

do_job([0,2,4,0,6,8]) #XTE

do_job([8,0,0,6,2,4]) #EXT

do_job([0,0,8,2,6,4]) #TXE


#3. Two clusters and ring

do_job([8,4,8,2,2,0]) #ETX

do_job([8,2,0,4,2,8]) #XET

do_job([0,2,8,2,4,8]) #TXE







for permut in list(permutations(np.arange(5))):
    np.concatenate([np.array(permut)*20+20,[0]])


if __name__ == "__main__":
    num_x, num_y = 15, 15

    WAB_WB_space = np.linspace(0,2,50)
    WA_WB_space = np.linspace(0,2,50)
    Cs = np.zeros([WAB_WB_space.size,WA_WB_space.size,num_x,num_y])
    X,Y = np.meshgrid(WAB_WB_space,WA_WB_space,indexing="ij")
    inputs = np.array([X.ravel(), Y.ravel()]).T

    n_slurm_tasks = int(os.environ["SLURM_NTASKS"])
    client = Client(threads_per_worker=1, n_workers=n_slurm_tasks,memory_limit="1GB")
    lazy_results = []
    for inputt in inputs:
        lazy_result = dask.delayed(do_job)(inputt)
        lazy_results.append(lazy_result)
    results = dask.compute(*lazy_results)
    if not os.path.exists("results"):
        os.makedirs("results")
    np.savetxt("results/params_%d.txt"%int(sys.argv[1]),inputs)
    for i, result in enumerate(results):
        save_npz("results/%d_%d.npz"%(i,int(sys.argv[1])),csc_matrix(result))


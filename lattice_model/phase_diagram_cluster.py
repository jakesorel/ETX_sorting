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

def do_job(inputt):
    N_E, N_T, N_X, scaler = 6, 6, 0, 3
    division = False
    seed_type = "random_circle"
    medium_energy = 5e2
    num_x, num_y = 15, 15
    T_t = np.logspace(3,-1,5*10**4)
    WAB_WB, WA_WB = inputt
    WB = 1
    W_ET = WAB_WB * WB
    W_EE = WA_WB * WB
    W_TT = WB
    W0 = np.array([[W_EE, W_ET, 0],
                   [W_ET, W_TT, 0],
                   [0, 0, 0]])
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
    return C_end.astype(np.int64)


if __name__ == "__main__":
    num_x, num_y = 15, 15

    WAB_WB_space = np.linspace(0,2,50)
    WA_WB_space = np.linspace(0,2,50)
    Cs = np.zeros([WAB_WB_space.size,WA_WB_space.size,num_x,num_y])
    X,Y = np.meshgrid(WAB_WB_space,WA_WB_space,indexing="ij")
    inputs = np.array([X.ravel(), Y.ravel()]).T

    n_slurm_tasks = int(os.environ["SLURM_NTASKS"])
    client = Client(threads_per_worker=1, n_workers=n_slurm_tasks)
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


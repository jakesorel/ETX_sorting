import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
import dask
from dask.distributed import Client

"""
Axial and radial asymmetry 
"""

num_x,num_y = 15,15
WAB_WB_space = np.linspace(0,2,50)
WA_WB_space = np.linspace(0,2,50)
X,Y = np.meshgrid(WAB_WB_space,WA_WB_space,indexing="ij")
inputs = np.array([X.ravel(), Y.ravel()]).T
iter_ids = [0,1,2,3,4,6,7,8,9]
n_iter = len(iter_ids)

def polarity_displacement(C):
    x,y = np.where(C!=0)
    centre = np.array([x.mean(),y.mean()])
    polarity, displacement = np.zeros([2,2]),np.zeros(2)
    for i, j in enumerate([1,2]):
        x, y = np.where(C ==j)
        displ = np.array([x,y]).T - centre
        polarity[i] = np.mean(displ,axis=0)
        displacement[i] = np.mean(np.linalg.norm(displ,axis=1))
    x, y = np.where(C != 0)
    displT = np.array([x, y]).T - centre
    displacementT = np.mean(np.linalg.norm(displT, axis=1))
    pol = np.abs(polarity[0]).sum()/displacementT
    disp = np.diff(displacement)/displacementT
    return pol, disp

polmat,dispmat = np.zeros((n_iter,inputs.shape[0])),np.zeros((n_iter,inputs.shape[0]))


def makemats(i):
    polmat, dispmat = np.zeros(inputs.shape[0]), np.zeros(inputs.shape[0])
    for j in range(inputs.shape[0]):
        try:
            C = load_npz("lattice_model/results/%d_%d.npz"%(j,i)).todense()
            polmat[j], dispmat[j] = polarity_displacement(C)
        except FileNotFoundError:
            polmat[j], dispmat[j] = np.nan,np.nan
    return polmat.reshape(X.shape),dispmat.reshape(X.shape)


n_slurm_tasks = 8
client = Client(threads_per_worker=1, n_workers=n_slurm_tasks,memory_limit="1GB")
lazy_results = []
for i in iter_ids:
    lazy_result = dask.delayed(makemats)(i)
    lazy_results.append(lazy_result)
mats = dask.compute(*lazy_results)

polmat,dispmat = np.zeros((n_iter,X.shape[0],X.shape[1])),np.zeros((n_iter,X.shape[0],X.shape[1]))
for i in range(n_iter):
    polmat[i],dispmat[i] = mats[i]

def fl(X):
    return np.flip(X.T,axis=0)

from scipy.interpolate import bisplrep,bisplev
pol, disp = np.nanmean(polmat,axis=0),np.nanmean(dispmat,axis=0)
polsd, dispsd = np.nanstd(polmat,axis=0),np.nanstd(dispmat,axis=0)


fig, ax = plt.subplots(1,2,figsize=(5,3))

extent = [WA_WB_space.min(),WA_WB_space.max(),WAB_WB_space.min(),WAB_WB_space.max()]
ax[0].imshow(np.flip(pol,axis=1),vmax=np.percentile(pol,80),vmin=np.percentile(pol,20),
             extent=extent,cmap=plt.cm.plasma)
ax[1].imshow(np.flip(disp,axis=1),
             extent=extent,cmap=plt.cm.viridis)
ax[0].set(ylabel=r"$W_{AB}/W_{BB}$",xlabel=r"$W_{AA}/W_{BB}$")
ax[1].set(xlabel=r"$W_{AA}/W_{BB}$")
ax[1].axes.get_yaxis().set_visible(False)

sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,norm=plt.Normalize(vmax=np.percentile(pol,80),vmin=np.percentile(pol,20)))
sm._A = []
cl = plt.colorbar(sm, ax=ax[0], pad=0.25, fraction=0.05, aspect=10, orientation="horizontal")#,ticks=np.linspace(0,1,2*N+1)[1::2])
cl.set_label("Axial asymmetry")
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,norm=plt.Normalize(vmax=disp.max(),vmin=disp.min()))
sm._A = []
cl = plt.colorbar(sm, ax=ax[1], pad=0.25, fraction=0.05, aspect=10, orientation="horizontal")#,ticks=np.linspace(0,1,2*N+1)[1::2])
cl.set_label("Radial asymmetry")
fig.subplots_adjust(top=0.9, bottom=0.25, left=0.2, right=0.8,wspace=0.05)
fig.savefig("axial vs radial asymmetry.pdf",dpi=300)



fig, ax = plt.subplots(1,3,figsize=(7.5,3))

extent = [WA_WB_space.min(),WA_WB_space.max(),WAB_WB_space.min(),WAB_WB_space.max()]
ax[0].imshow(np.flip(pol,axis=1),vmax=np.percentile(pol,80),vmin=np.percentile(pol,20),
             extent=extent,cmap=plt.cm.plasma)
ax[1].imshow(np.flip(disp,axis=1),
             extent=extent,cmap=plt.cm.viridis)
ax[2].imshow(np.flip(np.nan*disp,axis=1),
             extent=extent,cmap=plt.cm.viridis)
ax[0].set(ylabel=r"$W_{AB}/W_{BB}$",xlabel=r"$W_{AA}/W_{BB}$")
ax[1].set(xlabel=r"$W_{AA}/W_{BB}$")
ax[2].set(xlabel=r"$W_{AA}/W_{BB}$")

ax[1].axes.get_yaxis().set_visible(False)
ax[2].axes.get_yaxis().set_visible(False)

sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,norm=plt.Normalize(vmax=np.percentile(pol,80),vmin=np.percentile(pol,20)))
sm._A = []
cl = plt.colorbar(sm, ax=ax[0], pad=0.25, fraction=0.05, aspect=10, orientation="horizontal")#,ticks=np.linspace(0,1,2*N+1)[1::2])
cl.set_label("Axial asymmetry")
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,norm=plt.Normalize(vmax=disp.max(),vmin=disp.min()))
sm._A = []
cl = plt.colorbar(sm, ax=ax[1], pad=0.25, fraction=0.05, aspect=10, orientation="horizontal")#,ticks=np.linspace(0,1,2*N+1)[1::2])
cl.set_label("Radial asymmetry")
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,norm=plt.Normalize(vmax=disp.max(),vmin=disp.min()))
sm._A = []
cl = plt.colorbar(sm, ax=ax[2], pad=0.25, fraction=0.05, aspect=10, orientation="horizontal")#,ticks=np.linspace(0,1,2*N+1)[1::2])
cl.set_label("Radial asymmetry")
fig.subplots_adjust(top=0.9, bottom=0.25, left=0.2, right=0.8,wspace=0.05)
fig.savefig("axial vs radial asymmetry 3.pdf",dpi=300)


"""
Make some plots of case studies
"""


WAB_WB_space = np.linspace(0,2,50)
WA_WB_space = np.linspace(0,2,50)
X,Y = np.meshgrid(WAB_WB_space,WA_WB_space,indexing="ij")
inputs = np.array([X.ravel(), Y.ravel()]).T

# def get_C(x,y,j):
#     i = np.where((inputs[:,0]==WAB_WB_space[x])&(inputs[:,1]==WA_WB_space[y]))[0][0]
#     C = load_npz("lattice_model/results/%d_%d.npz"%(i,j)).todense()
#     return C
#
# plt.imshow(get_C(0,25,2))
# plt.show()

def get_C(i,j):
    C = load_npz("lattice_model/results/%d_%d.npz"%(j,i)).todense()
    return C

from lattice_model.self_organisation import ETX

etx = ETX()
for j in range(inputs.shape[0]):
    etx.plot_save(get_C(0,j),file_name=j,xlim=(0,50),ylim=(0,50))
    plt.close("all")



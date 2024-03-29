import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import animation, cm
# try:
from sim_analysis import Graph
# except ModuleNotFoundError:
#     from lattice_model.sim_analysis import Graph
import cv2
import os
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform,cdist
from numba import jit

class ETX:
    def __init__(self):
        self.x,self.y,self.X,self.Y = [],[],[],[]
        self.W = []
        self.C0 = []
        self.T_t = []
        self.t_span = []
        self.n_save = []
        self.t_save = [None]
        self.C_save = []
        self.n_print = None
        self.t_print = [None]
        self.division_time = []
        self.division_SD = []
        self.Timer = []
        self.xy_clls = []
        self.edge_mask = []
        self.Timer_edge_mask = []
        self.E = []
        self.well = []
        self.boundary = []

        self.valid_moves = np.array(np.meshgrid(np.arange(-1,2),np.arange(-1,2))).reshape(2,9)
        self.valid_moves = self.valid_moves[:,np.sum(self.valid_moves**2,axis=0)!=0]

        self.ID0 = []
        self.ID = []
        self.ID_save = []
        self.E_mat = []
        self.E_mat_dynamic = []
        self.adjacency_time = []
        self.t_crit, self.f_fin,self.t_n = [],[],[]
        self.dictionary = []

        self.E_t = []
        self.cell_number = []
        self.cell_number_t = []
        self.subpopulation_number = []
        self.subpopulation_number_t = []
        self.average_self_self = []
        self.average_self_self_t = []
        self.eccentricity = []
        self.eccentricity_t = []
        self.angular_distribution = []
        self.angular_polarity = []
        self.angular_polarity_t  = []

        self.t_equilib = []
        self.XEN_external_timescale = []


    def generate_lattice(self,num_x=100,num_y=100):
        """Generates a square lattice, with dimensions (num_x,num_y)"""
        self.x,self.y = np.arange(num_x),np.arange(num_y)
        self.X,self.Y = np.meshgrid(self.x,self.y,indexing="ij")

    def define_interaction_energies(self,W0 = None,boundary_scaler=0,sigma0=None):
        """Defines interaction energies.

        W0 is a 3x3 symmetric matrix defining the (positive) interaction energies of ES,TS,XEN

        boundary_scaler is added to W0 effectively defining the penalty for swapping with a medium element"""
        W = np.zeros([4,4])
        W[1:,1:] = boundary_scaler+W0
        self.medium_energy = boundary_scaler
        self.W = W
        if sigma0 is None:
            sigma0 = np.zeros_like(W)
        self.sigma0 = sigma0
        unique_IDs = np.arange(self.dictionary.shape[0])
        N = unique_IDs.size
        E_mat = np.zeros([N, N])
        E_mat0 = np.zeros([N, N])
        for id_i in [1, 2, 3]:
            for id_j in [1, 2, 3]:
                E_mat[id_i::3, id_j::3] = -np.random.normal(self.W[id_i, id_j], self.sigma0[id_i, id_j],
                                                               (int((N - 1) / 3), int((N - 1) / 3)))
                E_mat0[id_i::3, id_j::3] = -self.W[id_i, id_j]
        self.E_mat0 = E_mat0
        self.E_mat = E_mat
        self.adjacency_time = np.zeros_like(E_mat)

    def C0_generator_circle(self,N_E,N_T,N_X):
        """Generates the element matrix C at time=0: i.e. C0

        N_E, N_T,N_X defines the (approximate) number of cells of ES,TS and XEN at t=0

        The model is initialised with a fixed seed number for each cell type arranged in a pseudo-circular configuration,
         with the rest of the elements in the lattice being assigned medium identity. To avoid `gaps' in the initial
         condition, the seed numbers are used to estimate a circular radius
         ($r = \sqrt{\frac{N_{XEN}+N_{TS} + N_{ES}}{\pi}}$)
         which is then used to generate an enclosing square box, within the larger domain. Random elements within the
         box are assigned with probabilities proportional to the fraction of each of the seed numbers. Then a
         circular mask is applied to this randomly seeded box. This gives approximately the same number of each of
         the cell types as the values prescribed, but forces a filled circular geometry. """

        C0 = np.zeros_like(self.X)
        N = C0.size

        x0,y0 = self.X.shape[0]/2,self.X.shape[1]/2
        N_c0 = N_X + N_E + N_T
        r1 = np.sqrt((N_c0)/np.pi)
        circ = ((self.X+0.5-x0)**2 + (self.Y+0.5-y0)**2 > r1**2)
        circ_cell = ~circ

        box = ~((self.X<x0+r1)&(self.X>x0-r1)&(self.Y<y0+r1)&(self.Y>y0-r1))
        box_n = np.sum(~box).astype(int)

        N_t, N_e = int((N_T/N_c0)*box_n),int((N_E/N_c0)*box_n)
        if N_X!=0:
            N_x = box_n - N_t - N_e
        else:
            N_x = 0

        for k in range(N_x):
            i,j = np.random.randint(0,self.x.size),np.random.randint(0,self.y.size)
            while box[i,j] !=0:
                i, j = np.random.randint(0, self.x.size), np.random.randint(0, self.y.size)
            box[i,j] = 3
            C0[i,j] = 3

        for k in range(N_t):
            i,j = np.random.randint(0,self.x.size),np.random.randint(0,self.y.size)
            while box[i,j] !=0:
                i, j = np.random.randint(0, self.x.size), np.random.randint(0, self.y.size)
            box[i,j] = 2
            C0[i,j] = 2

        for k in range(N_e):
            i,j = np.random.randint(0,self.x.size),np.random.randint(0,self.y.size)
            while box[i,j] !=0:
                i, j = np.random.randint(0, self.x.size), np.random.randint(0, self.y.size)
            box[i,j] = 1
            C0[i,j] = 1

        C0 = C0*circ_cell
        return C0


    def C0_generator_random(self,N_E,N_T,N_X):
        x_clls, y_clls = np.where(~self.well)
        C0 = np.zeros_like(self.X)
        for l in range(N_E):
            k = int(np.random.random()*x_clls.size)
            i,j = x_clls[k],y_clls[k]
            C0[i,j] = 1
            x_clls,y_clls = np.delete(x_clls,k),np.delete(y_clls,k)
        for l in range(N_T):
            k = int(np.random.random()*x_clls.size)
            i,j = x_clls[k],y_clls[k]
            C0[i,j] = 2
            x_clls,y_clls = np.delete(x_clls,k),np.delete(y_clls,k)
        for l in range(N_X):
            k = int(np.random.random()*x_clls.size)
            i,j = x_clls[k],y_clls[k]
            C0[i,j] = 3
            x_clls,y_clls = np.delete(x_clls,k),np.delete(y_clls,k)
        return C0


    def C0_generator_random_circle(self,N_E,N_T,N_X):
        x0,y0 = self.X.shape[0]/2,self.X.shape[1]/2
        N_c0 = N_X + N_E + N_T
        r1 = int(np.ceil(np.sqrt((N_c0)/np.pi)))
        circ = ((self.X+0.5-x0)**2 + (self.Y+0.5-y0)**2 > r1**2)
        x_clls, y_clls = np.where(~circ)
        C0 = np.zeros_like(self.X)
        for l in range(N_E):
            k = int(np.random.random()*x_clls.size)
            i,j = x_clls[k],y_clls[k]
            C0[i,j] = 1
            x_clls,y_clls = np.delete(x_clls,k),np.delete(y_clls,k)
        for l in range(N_T):
            k = int(np.random.random()*x_clls.size)
            i,j = x_clls[k],y_clls[k]
            C0[i,j] = 2
            x_clls,y_clls = np.delete(x_clls,k),np.delete(y_clls,k)
        for l in range(N_X):
            k = int(np.random.random()*x_clls.size)
            i,j = x_clls[k],y_clls[k]
            C0[i,j] = 3
            x_clls,y_clls = np.delete(x_clls,k),np.delete(y_clls,k)
        return C0

    def make_C0(self,N_E,N_T,N_X,scaler,seed_type="circle"):
        """Generates the C0 matrix, using the C0_generator function

        N_E, N_T,N_X defines the (approximate) number of cells of ES,TS and XEN at t=0

        scaler will be used to multiply the values assigned for N_E, N_T, N_X"""

        N_X = int(N_X*scaler)
        N_E = int(N_E*scaler)
        N_T = int(N_T*scaler)
        self.N_E, self.N_X, self.N_T, self.N = N_E, N_X, N_T, N_E+N_X+N_T
        if seed_type == "circle":
            C0 = self.C0_generator_circle(N_E,N_T,N_X)
        if seed_type == "random":
            C0 = self.C0_generator_random(N_E,N_T,N_X)
        if seed_type == "random_circle":
            C0 = self.C0_generator_random_circle(N_E,N_T,N_X)
        self.C0 = C0
        return C0

    def make_ID0(self,division=False):
        """Each cell is given a unique ID. The spatial configuration of these cells is kept track of in the ID0 matrix.

        A simple convention to make handing data easier is that ES, TS, and XEN cells are alternately assigned.
        I.e. mod(id-1,3) = type {-1 as need to account for the fact that 0 is assigned for medium, both for id and type}
        """
        if division is False:
            ID0 = np.zeros_like(self.C0)
            dict_len = np.max([np.sum(self.C0==id) for id in [1,2,3]])*3+1
            dictionary = np.zeros([dict_len])
            for id in [1,2,3]:
                x_clls,y_clls = np.where(self.C0==id)
                xy_clls = np.array([x_clls,y_clls]).T
                for i, ij in enumerate(xy_clls):
                    ID0[ij[0],ij[1]] = 3*i+id
                    dictionary[3*i+id] = self.C0[ij[0],ij[1]]
        if division is True:
            ID0 = np.zeros_like(self.C0)
            for id in [1,2,3]:
                x_clls,y_clls = np.where(self.C0==id)
                xy_clls = np.array([x_clls,y_clls]).T
                for i, ij in enumerate(xy_clls):
                    ID0[ij[0],ij[1]] = 3*i+id

            dict_len = (np.max([np.sum(self.C0==id) for id in [1,2,3]])*3)*int((2**(self.T_t.size/self.division_time+1)))+1 #this sets an upper bound on the number of cells at the end of the simulation
            dictionary = np.zeros([dict_len])
            dictionary[1::3],dictionary[2::3],dictionary[3::3] = 1,2,3
        self.dictionary = dictionary.astype(int)
        self.ID0 = ID0
        return ID0

    def define_running_params(self,T_t = np.repeat(0.25,1*10**5)):
        """Defines the temperature regime across the simulation.

        T_t: Temperature at each time-step. Length of T_t defines the number of iterations"""
        self.T_t = T_t
        self.t_span = np.arange(T_t.size)

    def define_saving_params(self,n_save=200):
        """n_save is the number of iterations that the simulation output will be saved
        Generates ID_save, an array that logs ID at each of these time-points.
        Generates t_save, a vector defining the time-points of saving"""
        self.t_save = self.t_span[::int(np.ceil(self.t_span.size/n_save))]
        self.n_save = self.t_save.size
        self.ID_save = np.zeros([n_save,self.ID0.shape[0], self.ID0.shape[1]])

    def define_printing_params(self,n_print=20):
        """n_print is the number of iterations the simulation will print the % progress"""
        self.n_print = n_print
        self.t_print = self.t_span[::int(np.ceil(self.t_span.size/n_print))]

    def define_division_params(self,division_number=4,mean_over_SD=8,Timer=None):
        """Sets up division parameters. Generates a matrix Timer of the same shape as C.

        Timer keeps track of the time until the next division.

        Timer is initially seeded with a uniform distribution by default, preventing divisions in the first mean/8
        iterations. But can be customised (requires that dim(Timer) = dim(C)

        division_number defines the number of divisions that each cell should take on average across the simulation

        mean_over_SD defines the ratio of the mean to the standard deviation of division times. (After the first
        division, division time follows a Normal distribution, with means and SDs prescribed here).
        """

        self.division_time = self.T_t.size / division_number
        self.division_SD = self.division_time / mean_over_SD
        if Timer is None:
            Timer = np.random.uniform(self.division_time / 8, self.division_time,self.X.shape)
        Timer[np.where(self.C0 == 0)] = np.inf
        self.Timer = Timer

    def Energy(self,ID,i,j,id):
        """Defines the energy of a given cell, defined by its interaction with the 8 surrounding
        (i.e. inc. diagonals) cells

        ID is the identity matrix matrix

        i,j defines the cell in question; coords within the matrix

        id defines the identity of the element"""
        ID2 = ID.copy()
        ID2[i,j] = -1
        di = (i>0)*(i-1), (i<(self.x.size-1))*(i+2) + (i==(self.x.size-1))*(i+1)
        dj = (j>0)*(j-1), (j<(self.y.size-1))*(j+2) + (j==(self.y.size-1))*(j+1)
        bordering_cells = ID2[di[0]:di[1],dj[0]:dj[1]].flatten()
        E = 0
        for i in bordering_cells:
            if i != -1:
                E -= self.E_mat[i,id]
        return E



    def dEnergy(self,ID, i, j, di, dj):
        """Defines the energy change associated with a swap.
        i,j define the matrix position of the element in question
        di,dj define the direction of the putatively swapped cell (di,dj = -1,0,1)

        NB: verbose code, but optimised for efficiency"""
        return dEnergy(self.E_mat, ID, i, j, di, dj)
        # dE = 0
        # E_mat= self.E_mat
        # II,JJ = ID[i,j], ID[i+di,j+dj]
        # if di == 0:
        #     for I in range(i - 1, i + 2):
        #         dE -= E_mat[JJ,ID[I, j + 2 * dj]]
        #         dE -= E_mat[II, ID[I, j -dj]]
        #         dE += E_mat[II, ID[I, j + 2 * dj]]
        #         dE += E_mat[JJ, ID[I, j - dj]]
        #
        # elif dj == 0:
        #     for J in range(j - 1, j + 2):
        #         dE -= E_mat[JJ, ID[i + 2 * di, J]]
        #         dE -= E_mat[II, ID[i - di, J]]
        #         dE += E_mat[II, ID[i + 2 * di, J]]
        #         dE += E_mat[JJ, ID[i - di, J]]
        # else:
        #     ID_flat1 = ID[i - 1:i + 2, j - 1:j + 2].ravel()
        #     ID_flat2 = ID[i + di - 1:i + di + 2, j + dj - 1:j + dj + 2].ravel()
        #     if (di == 1) & (dj == 1):
        #         ids1 = ID_flat1[[0, 1, 2, 3, 6]]
        #         ids2 = ID_flat2[[2, 5, 6, 7, 8]]
        #
        #     if (di == -1) & (dj == -1):
        #         ids1 = ID_flat1[[2, 5, 6, 7, 8]]
        #         ids2 = ID_flat2[[0, 1, 2, 3, 6]]
        #
        #     if (di == 1) & (dj == -1):
        #         ids1 = ID_flat1[[0, 1, 2, 5, 8]]
        #         ids2 = ID_flat2[[0, 3, 6, 7, 8]]
        #
        #     if (di == -1) & (dj == 1):
        #         ids1 = ID_flat1[[0, 3, 6, 7, 8]]
        #         ids2 = ID_flat2[[0, 1, 2, 5, 8]]
        #
        #     for id in ids1:
        #         dE -= E_mat[II, id]
        #         dE += E_mat[JJ, id]
        #     for id in ids2:
        #         dE -= E_mat[JJ, id]
        #         dE += E_mat[II, id]
        # return 2*dE #count both directions of an interaction

    def get_adjacency(self,ID):
        A_mat = np.zeros_like(self.E_mat)
        adj_array = np.stack([np.roll(ID,1,axis=0),
                      np.roll(ID,-1,axis=0),
                      np.roll(ID,1,axis=1),
                      np.roll(ID,-1,axis=1),
                      np.roll(np.roll(ID,1,axis=1),1,axis=0),
                      np.roll(np.roll(ID,-1,axis=1),1,axis=0),
                      np.roll(np.roll(ID,1,axis=1),-1,axis=0),
                      np.roll(np.roll(ID,-1,axis=1),-1,axis=0)])
        rw,cl = np.where(ID!=0)
        A_mat[ID[rw,cl],adj_array[:, rw, cl]] = 1
        A_mat[0] = 0
        A_mat[:,0] = 0
        return A_mat

    def adjacency_timer(self,ID):
        """Cells build up +1 adjacency time with each timestep of contact. But this reverts to 0 when contact lost"""
        self.adjacency_time += 1
        self.adjacency_time = self.adjacency_time*self.get_adjacency(ID)

    def set_dynamic_params(self,t_crit,f_fin,t_n):
        self.t_crit, self.f_fin,self.t_n = t_crit, f_fin,t_n

    def update_E_mat_dynamic(self):
        E_mat_additional = np.zeros_like(self.E_mat)
        rw, cl = np.where(self.adjacency_time!=0)
        t = self.adjacency_time[rw,cl]
        E_mat_additional[rw,cl] = self.f_fin/(1+(self.t_crit/t)**self.t_n)
        self.E_mat_dynamic = (self.E_mat-self.medium_energy)*(1 + E_mat_additional)+self.medium_energy

    def update_T_dynamic(self):
        T_mat_additional = np.zeros_like(self.E_mat)
        rw, cl = np.where(self.adjacency_time!=0)
        t = self.adjacency_time[rw,cl]
        T_mat_additional[rw,cl] = self.f_fin/(1+(self.t_crit/t)**self.t_n)
        self.dynT = np.sum(T_mat_additional,axis=0)/8



    def dEnergy_dyn(self,ID, i, j, di, dj):
        """Defines the energy change associated with a swap.
        i,j define the matrix position of the element in question
        di,dj define the direction of the putatively swapped cell (di,dj = -1,0,1)

        NB: verbose code, but optimised for efficiency"""
        dE = 0
        E_mat= self.E_mat_dynamic
        II,JJ = ID[i,j], ID[i+di,j+dj]
        if di == 0:
            for I in range(i - 1, i + 2):
                dE -= E_mat[JJ,ID[I, j + 2 * dj]]
                dE -= E_mat[II, ID[I, j -dj]]
                dE += E_mat[II, ID[I, j + 2 * dj]]
                dE += E_mat[JJ, ID[I, j - dj]]

        elif dj == 0:
            for J in range(j - 1, j + 2):
                dE -= E_mat[JJ, ID[i + 2 * di, J]]
                dE -= E_mat[II, ID[i - di, J]]
                dE += E_mat[II, ID[i + 2 * di, J]]
                dE += E_mat[JJ, ID[i - di, J]]
        else:
            ID_flat1 = ID[i - 1:i + 2, j - 1:j + 2].ravel()
            ID_flat2 = ID[i + di - 1:i + di + 2, j + dj - 1:j + dj + 2].ravel()
            if (di == 1) & (dj == 1):
                ids1 = ID_flat1[[0, 1, 2, 3, 6]]
                ids2 = ID_flat2[[2, 5, 6, 7, 8]]

            if (di == -1) & (dj == -1):
                ids1 = ID_flat1[[2, 5, 6, 7, 8]]
                ids2 = ID_flat2[[0, 1, 2, 3, 6]]

            if (di == 1) & (dj == -1):
                ids1 = ID_flat1[[0, 1, 2, 5, 8]]
                ids2 = ID_flat2[[0, 3, 6, 7, 8]]

            if (di == -1) & (dj == 1):
                ids1 = ID_flat1[[0, 3, 6, 7, 8]]
                ids2 = ID_flat2[[0, 1, 2, 5, 8]]

            for id in ids1:
                dE -= E_mat[II, id]
                dE += E_mat[JJ, id]
            for id in ids2:
                dE -= E_mat[JJ, id]
                dE += E_mat[II, id]
        return 2*dE #count both directions of an interaction


    def dEnergy_dynamic(self,ID, i, j, di, dj,alpha=0.2):
        """Defines the energy change associated with a swap.
        i,j define the matrix position of the element in question
        di,dj define the direction of the putatively swapped cell (di,dj = -1,0,1)

        NB: verbose code, but optimised for efficiency"""
        dE = 0
        E_mat= self.E_mat
        II,JJ = ID[i,j], ID[i+di,j+dj]
        if di == 0:
            for I in range(i - 1, i + 2):
                dE -= E_mat[JJ,ID[I, j + 2 * dj]]
                dE -= E_mat[II, ID[I, j -dj]]
                dE += E_mat[II, ID[I, j + 2 * dj]]
                dE += E_mat[JJ, ID[I, j - dj]]

        elif dj == 0:
            for J in range(j - 1, j + 2):
                dE -= E_mat[JJ, ID[i + 2 * di, J]]
                dE -= E_mat[II, ID[i - di, J]]
                dE += E_mat[II, ID[i + 2 * di, J]]
                dE += E_mat[JJ, ID[i - di, J]]
        else:
            ID_flat1 = ID[i - 1:i + 2, j - 1:j + 2].ravel()
            ID_flat2 = ID[i + di - 1:i + di + 2, j + dj - 1:j + dj + 2].ravel()
            if (di == 1) & (dj == 1):
                ids1 = ID_flat1[[0, 1, 2, 3, 6]]
                ids2 = ID_flat2[[2, 5, 6, 7, 8]]

            if (di == -1) & (dj == -1):
                ids1 = ID_flat1[[2, 5, 6, 7, 8]]
                ids2 = ID_flat2[[0, 1, 2, 3, 6]]

            if (di == 1) & (dj == -1):
                ids1 = ID_flat1[[0, 1, 2, 5, 8]]
                ids2 = ID_flat2[[0, 3, 6, 7, 8]]

            if (di == -1) & (dj == 1):
                ids1 = ID_flat1[[0, 3, 6, 7, 8]]
                ids2 = ID_flat2[[0, 1, 2, 5, 8]]

            for id in ids1:
                dE -= E_mat[II, id]
                dE += E_mat[JJ, id]
            for id in ids2:
                dE -= E_mat[JJ, id]
                dE += E_mat[II, id]
        return 2*dE+self.dynamic_boost_value(ID,i,j)*alpha #count both directions of an interaction

    #
    #
    # def dynamic_boost_value(self,ID,i,j):
    #     ids = ID[i-1:i+2,j-1:j+2].ravel()
    #     id0 = ID[i,j]
    #     boost_value = 0
    #     for i, id in enumerate(ids):
    #         if i!=4:
    #             boost_value += self.E_mat[id0,id]
    #     return boost_value
    #
    # def dEnergy_dynamic(self,ID, i, j, di, dj,alpha=0.2):
    #     """Defines the energy change associated with a swap.
    #     i,j define the matrix position of the element in question
    #     di,dj define the direction of the putatively swapped cell (di,dj = -1,0,1)
    #
    #     NB: verbose code, but optimised for efficiency"""
    #     dE = 0
    #     E_mat= self.E_mat
    #     II,JJ = ID[i,j], ID[i+di,j+dj]
    #     if di == 0:
    #         for I in range(i - 1, i + 2):
    #             dE -= E_mat[JJ,ID[I, j + 2 * dj]]
    #             dE -= E_mat[II, ID[I, j -dj]]
    #             dE += E_mat[II, ID[I, j + 2 * dj]]
    #             dE += E_mat[JJ, ID[I, j - dj]]
    #
    #     elif dj == 0:
    #         for J in range(j - 1, j + 2):
    #             dE -= E_mat[JJ, ID[i + 2 * di, J]]
    #             dE -= E_mat[II, ID[i - di, J]]
    #             dE += E_mat[II, ID[i + 2 * di, J]]
    #             dE += E_mat[JJ, ID[i - di, J]]
    #     else:
    #         ID_flat1 = ID[i - 1:i + 2, j - 1:j + 2].ravel()
    #         ID_flat2 = ID[i + di - 1:i + di + 2, j + dj - 1:j + dj + 2].ravel()
    #         if (di == 1) & (dj == 1):
    #             ids1 = ID_flat1[[0, 1, 2, 3, 6]]
    #             ids2 = ID_flat2[[2, 5, 6, 7, 8]]
    #
    #         if (di == -1) & (dj == -1):
    #             ids1 = ID_flat1[[2, 5, 6, 7, 8]]
    #             ids2 = ID_flat2[[0, 1, 2, 3, 6]]
    #
    #         if (di == 1) & (dj == -1):
    #             ids1 = ID_flat1[[0, 1, 2, 5, 8]]
    #             ids2 = ID_flat2[[0, 3, 6, 7, 8]]
    #
    #         if (di == -1) & (dj == 1):
    #             ids1 = ID_flat1[[0, 3, 6, 7, 8]]
    #             ids2 = ID_flat2[[0, 1, 2, 5, 8]]
    #
    #         for id in ids1:
    #             dE -= E_mat[II, id]
    #             dE += E_mat[JJ, id]
    #         for id in ids2:
    #             dE -= E_mat[JJ, id]
    #             dE += E_mat[II, id]
    #     return 2*dE+self.dynamic_boost_value(ID,i,j)*alpha #count both directions of an interaction
    #


    def E_tot(self,ID):
        """Defines the total energy of a given configuration (C).
        This is the sum of the individual energies"""
        E = 0
        for i in range(ID.shape[0]):
            for j in range(ID.shape[1]):
                E += self.Energy(ID,i,j,ID[i,j])
        return E

    def kth_diag_indices(self,a, k):
        """
        Finds the indices of the offset diagonal of a matrix.

        From https://stackoverflow.com/questions/10925671/numpy-k-th-diagonal-indices

        a is the matrix

        k is the offset
        (positively offset in the j direction & negatively offset in the i direction by magnitude k)
        """
        rows, cols = np.diag_indices_from(a)
        if k < 0:
            return rows[-k:], cols[:k]
        elif k > 0:
            return rows[:-k], cols[k:]
        else:
            return rows, cols

    def divider(self,X,i,j,dir,N_cells=None):
        """Performs a division at element i,j. Returns a matrix with the element divided.

        X is the matrix to undergo a division

        i,j defines the element to be divided (i.e. multiplied)

        dir defines the direction of the division:
            0: down
            1: up
            2: left
            3: right
            4: down-right
            5: up-left
            6: down-left
            7: up-right"""
        XX = X.copy()
        if dir == 0:
            XX[i + 1:, j] = XX[i:-1, j]
        if dir == 1:
            XX[:i, j] = XX[1:i + 1, j]
        if dir == 2:
            XX[i, j + 1:] = XX[i, j:-1]
        if dir == 3:
            XX[i, :j] = XX[i, 1:j + 1]
        if dir == 4:
            k = j - i
            l = np.min([i,j])
            rw, cl = self.kth_diag_indices(XX,k)
            XX[rw[l+1:],cl[l+1:]] = XX[rw,cl][l:-1]
        if dir == 5:
            k = j - i
            l = np.min([i, j])
            rw, cl = self.kth_diag_indices(XX, k)
            XX[rw[:l],cl[:l]] = XX[rw,cl][1:l+1]
        if dir == 6:  #
            XXX = np.flip(XX,axis=1)
            jj = XX.shape[1]-j - 1
            k = jj - i
            l = np.min([i, jj])
            rw, cl = self.kth_diag_indices(XXX, k)
            XXX[rw[l + 1:], cl[l + 1:]] = XXX[rw, cl][l:-1]
            XX = np.flip(XXX,axis=1)
        if dir == 7:  # up-right
            XXX = np.flip(XX,axis=1)
            jj = XX.shape[1]-j - 1
            k = jj - i
            l = np.min([i, jj])
            rw, cl = self.kth_diag_indices(XXX, k)
            XXX[rw[:l],cl[:l]] = XXX[rw,cl][1:l+1]
            XX = np.flip(XXX,axis=1)
        if N_cells is not None:
            cid = self.dictionary[XX[i,j]]
            XX[i,j] = N_cells[cid-1] * 3 + cid
            N_cells[cid-1] = N_cells[cid-1]+1
            return XX,N_cells
        else:
            return XX


    def generate_iijj_from_dir(self,i,j,dir):
        """Finds the new daughter cell post-division.

        i,j is defines the location of the mother cell

        dir defines the division direction (see above)"""
        if dir==0:
            return i+1,j
        if dir==1:
            return i-1,j
        if dir==2:
            return i,j+1
        if dir==3:
            return i,j-1
        if dir==4:
            return i+1,j+1
        if dir==5:
            return i-1,j-1
        if dir==6:
            return i+1, j-1
        if dir==7:
            return i-1,j+1

    def swapper(self,X,C,i,j,ii,jj,xy_clls=False):
        """Performs a swapping event.

        X is the matrix to undergo a swap.

        i,j is the cell chosen by the MH algorithm

        ii,jj is the cell that is going to be swapped with cell (i,j)

        xy_clls keeps track of the non-medium elements (i.e. cells). Updated in cases of a medium-cell swap

        C is the configuration matrix. Used to identify changes to xy_clls when medium-cell swaps occur
        """
        XX = X.copy()
        XX[ii, jj] = X[i, j].copy()
        XX[i, j] = X[ii, jj].copy()
        if xy_clls is not False:
            if C[i,j] ==0:
                id = np.where(np.sum(np.absolute(xy_clls-np.array([ii,jj])),axis=1)==0)[0][0]
                xy_clls = np.delete(xy_clls,id,axis=0)
                xy_clls = np.vstack([xy_clls, np.array([i, j])])
            if C[ii,jj]==0:
                id = np.where(np.sum(np.absolute(xy_clls - np.array([i, j])), axis=1) == 0)[0][0]
                xy_clls = np.delete(xy_clls, id, axis=0)
                xy_clls = np.vstack([xy_clls, np.array([ii, jj])])
            return XX,xy_clls
        else:
            return XX

    def number_of_cells(self,C):
        """Finds the number of ES, TS, and XEN cells"""
        return np.array([np.sum(C == 1), np.sum(C == 2), np.sum(C == 3)])

    def initialise_simulation(self):
        """Generates the initial cell list (xy_clls).

        And generates the boundary masks that remove cells that cross the outside of the matrix"""
        x_clls,y_clls = np.where(self.C0!=0)
        self.xy_clls = np.array([x_clls,y_clls]).T

        edge_mask = np.ones_like(self.C0)
        edge_mask[0:2],edge_mask[-2:] = 0,0
        edge_mask[:,0:2],edge_mask[:,-2:] = 0,0
        self.edge_mask = edge_mask

        Timer_edge_mask = np.zeros_like(self.C0)
        Timer_edge_mask[0:2],Timer_edge_mask[-2:] = np.inf,np.inf
        Timer_edge_mask[:,0:2],Timer_edge_mask[:,-2:] = np.inf,np.inf
        self.Timer_edge_mask = Timer_edge_mask

    def boundary_definition(self,well=None):
        if well is None:
            x0, y0 = self.X.shape[0] / 2, self.X.shape[1] / 2
            r1 = int(self.X.shape[0] / 2) - 1
            well = ((self.X + 0.5 - x0) ** 2 + (self.Y + 0.5 - y0) ** 2 > r1 ** 2)
        x_clls, y_clls = np.where(~well)
        boundary = np.zeros_like(self.X)
        for x, i in enumerate(x_clls):
            j = y_clls[x]
            neighbours = self.neighbourhood_possibility(well, i, j)
            if np.sum(neighbours) != neighbours.size:
                boundary[i, j] = 1
        self.boundary = boundary
        self.well = well

    def neighbourhood_possibility(self,well, i, j):
        """
                0: down
                1: up
                2: left
                3: right
                4: down-right
                5: up-left
                6: down-left
                7: up-right
        """
        sample = ~well[i - 1:i + 2, j - 1:j + 2].ravel()
        return sample[[7, 1, 3, 6, 8, 0, 6, 2]]

    def boundary_valid_didj(self,well, i, j, di, dj):
        sample = ~well[i - 1:i + 2, j - 1:j + 2]
        return sample[1 + di, 1 + dj]


    def self_contacts(self,C):
        udlr = (C==np.roll(C,1,axis=0))*1.0+(C==np.roll(C,1,axis=1))*1.0+(C==np.roll(C,-1,axis=0))*1.0+(C==np.roll(C,-1,axis=1))*1.0
        diags = np.zeros_like(udlr)
        diags[:-1,:-1] += 1.0*(C[:-1,:-1] == C[1:,1:])
        diags[1:, 1:] += 1.0*(C[:-1,:-1] == C[1:,1:])
        diags[1:, :-1] += 1.0 * (C[1:, :-1] == C[:-1, 1:])
        diags[:-1, 1:] += 1.0 * (C[1:, :-1] == C[:-1, 1:])
        contacts = udlr+diags
        return contacts

    def self_contact_ij(self,C,i,j):
        contacts = np.sum(C[i-1:i+2,j-1:j+2]==C[i,j])-1
        return contacts

    def contact_integration(self,C,V,k_gain,k_loss):
        """For now, consider only self-self interactions. EXPAND, using W0 in the future

        propose that dtV = k_gain*n_contacts - k_loss

        for conv., dt = 1"""
        contacts = self.self_contacts(C)
        V = V+k_gain*contacts - V*k_loss
        return V

    def ij_move(self):
        return self.valid_moves[:,int(np.random.random()*8)]

    def valid_move(self,i,j,di,dj):
        return (di + i >= 0) & (di + i < self.x.size - 1) & (dj + j >= 0) & (dj + j < self.y.size - 1)

    def perform_simulation(self,swap_rate=10,division=False,end_only=False):
        """Performs Metropolis-Hastings. Starting with C0, iterates a random cell selection and putative
       swapping procedure. When cells reach their division time, they undergo divisions

        swap_rate entails the number of cells (as a  proportion of the total number of cells of the embryoid as a whole)
            that are selected in each iteration.
            When swap_rate = 1, N cells are selected with each iteration, where N is the number of cells
            (Note that cell selection is still stochastic i.e. swap_rate =1 does NOT mean that every cell attempts
            to undergo a swap

        if division is False: only swapping is considered. swap_rate term is ignored, with one time-point
        considering one swap"""

        ID = self.ID0
        xy_clls = self.xy_clls
        N_cells = self.number_of_cells(self.C0)
        if division is True:
            Timer = self.Timer
            for t, T in enumerate(self.T_t):

                #1. Divide
                Timer = Timer - 1
                if np.sum(Timer < 0)!=0:
                    i_div, j_div = np.where(Timer<0)
                    while i_div.size !=0:
                        id = np.random.randint(i_div.size)
                        i,j = i_div[id],j_div[id]
                        if self.boundary[i,j]:
                            poss_dirs = np.where(self.neighbourhood_possibility(self.well, i, j) == 1)[0]
                            dir = poss_dirs[int(poss_dirs.size*np.random.random())]
                        else:
                            dir = int(8*np.random.random())
                        ID,N_cells = self.divider(ID,i, j, dir,N_cells=N_cells)
                        Timer = self.divider(Timer,i, j, dir)
                        Timer[i,j] = np.random.normal(self.division_time,self.division_SD)
                        Timer[self.generate_iijj_from_dir(i,j,dir)] = np.random.normal(self.division_time,self.division_SD)
                        i_div, j_div = np.where(Timer < 0)
                    x_clls, y_clls = np.where(ID != 0)
                    xy_clls = np.array([x_clls, y_clls]).T


                for n in range(int(xy_clls.shape[0]/swap_rate)): #approximately scale the rate of swapping with embryo size, while allowing for division to occur psuedo-simultaneously.
                    #2. Select a random cell
                    cll_id = int(np.floor(np.random.random() * xy_clls.shape[0]))
                    i, j = xy_clls[cll_id]

                    if self.boundary[i,j]:
                        # 3. Define a putative move
                        di, dj = self.ij_move()
                        while not (self.valid_move(i,j,di,dj)& self.boundary_valid_didj(self.well, i, j, di, dj)):  # in some cases, re-sample if point chosen not within graph
                            di, dj = self.ij_move()
                    else:
                        #3. Define a putative move
                        di,dj = self.ij_move()
                        while not self.valid_move(i,j,di,dj):#in some cases, re-sample if point chosen not within graph
                            di, dj = self.ij_move()
                    ii, jj = i+di, j+dj
                    dE = self.dEnergy(ID,i,j,di,dj)
                    if (dE < 0):
                        ID,xy_clls = self.swapper(ID,ID,i,j,ii,jj,xy_clls)
                        Timer = self.swapper(Timer,ID,i,j,ii,jj)
                    elif np.random.random()< np.exp(-dE/T):
                        ID,xy_clls = self.swapper(ID,ID,i,j,ii,jj,xy_clls)
                        Timer = self.swapper(Timer,ID,i,j,ii,jj)

                    self.XXX = ID.copy()


                if t in self.t_save:
                    self.ID_save[np.where(t==self.t_save)[0][0]] = ID
                if t in self.t_print:
                    print("%d %% completed"%(100*t/self.t_print[-1]))
            self.Timer = Timer
        if division is False:
            for t, T in enumerate(self.T_t):
                # 2. Select a random cell
                cll_id = int(np.floor(np.random.random() * xy_clls.shape[0]))
                i, j = xy_clls[cll_id]

                if self.boundary[i, j]:
                    # 3. Define a putative move
                    di, dj = self.ij_move()
                    while not (self.valid_move(i,j,di,dj)& self.boundary_valid_didj(self.well, i, j, di, dj)):  # in some cases, re-sample if point chosen not within graph
                        di, dj = self.ij_move()
                else:
                    # 3. Define a putative move
                    di, dj = self.ij_move()
                    while not self.valid_move(i,j,di,dj):  # in some cases, re-sample if point chosen not within graph
                        di, dj = self.ij_move()
                ii, jj = i + di, j + dj
                dE = self.dEnergy(ID, i, j, di, dj)
                if (dE < 0):
                    ID, xy_clls = self.swapper(ID, ID, i, j, ii, jj, xy_clls)
                elif np.random.random() < np.exp(-dE / T):
                    ID, xy_clls = self.swapper(ID, ID, i, j, ii, jj, xy_clls)
                if end_only is False:
                    if t in self.t_save:
                            self.ID_save[np.where(t == self.t_save)[0][0]] = ID
                    if t in self.t_print:
                        print("%d %% completed" % (100 * t / self.t_print[-1]))
        self.ID = ID
        self.ID_save[-1] = ID

        # self.E = self.E_tot(C)


    def perform_simulation_to_equilibrium(self,swap_rate=10,division=False,max_time=1e6,check_freq=1e2,track_XEN=False):
        """Performs Metropolis-Hastings. Starting with C0, iterates a random cell selection and putative
       swapping procedure. When cells reach their division time, they undergo divisions

        swap_rate entails the number of cells (as a  proportion of the total number of cells of the embryoid as a whole)
            that are selected in each iteration.
            When swap_rate = 1, N cells are selected with each iteration, where N is the number of cells
            (Note that cell selection is still stochastic i.e. swap_rate =1 does NOT mean that every cell attempts
            to undergo a swap

        if division is False: only swapping is considered. swap_rate term is ignored, with one time-point
        considering one swap

        Performs until equilibrium is reached"""

        ID = self.ID0
        xy_clls = self.xy_clls
        N_cells = self.number_of_cells(self.C0)
        if division is True:
            print("Division not configured for this yet")
            # Timer = self.Timer
            # t = 0
            # T = self.T_t[0]
            # subpopE,subpopT = 10,10 #seed with a number > 1
            # while t < max_time:
            #     while (subpopE>1)&(subpopT>1):
            #         #1. Divide
            #         Timer = Timer - 1
            #         if np.sum(Timer < 0)!=0:
            #             i_div, j_div = np.where(Timer<0)
            #             while i_div.size !=0:
            #                 id = np.random.randint(i_div.size)
            #                 i,j = i_div[id],j_div[id]
            #                 if self.boundary[i,j]:
            #                     poss_dirs = np.where(self.neighbourhood_possibility(self.well, i, j) == 1)[0]
            #                     dir = poss_dirs[int(poss_dirs.size*np.random.random())]
            #                 else:
            #                     dir = int(8*np.random.random())
            #                 ID,N_cells = self.divider(ID,i, j, dir,N_cells=N_cells)
            #                 Timer = self.divider(Timer,i, j, dir)
            #                 Timer[i,j] = np.random.normal(self.division_time,self.division_SD)
            #                 Timer[self.generate_iijj_from_dir(i,j,dir)] = np.random.normal(self.division_time,self.division_SD)
            #                 i_div, j_div = np.where(Timer < 0)
            #             x_clls, y_clls = np.where(ID != 0)
            #             xy_clls = np.array([x_clls, y_clls]).T
            #
            #
            #         for n in range(int(xy_clls.shape[0]/swap_rate)): #approximately scale the rate of swapping with embryo size, while allowing for division to occur psuedo-simultaneously.
            #             #2. Select a random cell
            #             cll_id = int(np.floor(np.random.random() * xy_clls.shape[0]))
            #             i, j = xy_clls[cll_id]
            #
            #             if self.boundary[i,j]:
            #                 # 3. Define a putative move
            #                 di, dj = int(np.random.random() * 3) - 1, int(np.random.random() * 3) - 1
            #                 while not (di + i >= 0) & (di + i < self.x.size) & (dj + j >= 0) & (dj + j < self.y.size) & (
            #                         (di != 0) or (dj != 0))&self.boundary_valid_didj(self.well,i,j,di,dj):  # in some cases, re-sample if point chosen not within graph
            #                     di, dj = int(np.random.random() * 3) - 1, int(np.random.random() * 3) - 1
            #             else:
            #                 #3. Define a putative move
            #                 di,dj = int(np.random.random()*3) - 1, int(np.random.random()*3) - 1
            #                 while not (di+i>=0)&(di+i<self.x.size)&(dj+j>=0)&(dj+j<self.y.size)&((di!=0)or(dj!=0)):#in some cases, re-sample if point chosen not within graph
            #                     di, dj = int(np.random.random()*3) - 1,int(np.random.random()*3) - 1
            #             ii, jj = i+di, j+dj
            #             dE = self.dEnergy(ID,i,j,di,dj)
            #             if (dE < 0):
            #                 ID,xy_clls = self.swapper(ID,ID,i,j,ii,jj,xy_clls)
            #                 Timer = self.swapper(Timer,ID,i,j,ii,jj)
            #             elif np.random.random()< np.exp(-dE/T):
            #                 ID,xy_clls = self.swapper(ID,ID,i,j,ii,jj,xy_clls)
            #                 Timer = self.swapper(Timer,ID,i,j,ii,jj)
            #
            #
            #         if t in self.t_save:
            #             self.ID_save[np.where(t==self.t_save)[0][0]] = ID
            #         if t in self.t_print:
            #             print("%d %% completed"%(100*t/self.t_print[-1]))
            #         t +=1
            # self.Timer = Timer
        if division is False:
            t = 0
            subpopE,subpopT = 10,10 #seed with a number > 1
            XEN_external=0
            XEN_external_timescale=max_time
            while (t < max_time)and((subpopE>1)or(subpopT>1)):
                for n in range(self.N):
                    T = self.T_t[t]
                    # 2. Select a random cell
                    cll_id = int(np.floor(np.random.random() * xy_clls.shape[0]))
                    i, j = xy_clls[cll_id]

                    if self.boundary[i, j]:
                        # 3. Define a putative move
                        di, dj = int(np.random.random() * 3) - 1, int(np.random.random() * 3) - 1
                        while not (di + i >= 0) & (di + i < self.x.size) & (dj + j >= 0) & (dj + j < self.y.size) & (
                                (di != 0) or (dj != 0)) & self.boundary_valid_didj(self.well, i, j, di,
                                                                                   dj):  # in some cases, re-sample if point chosen not within graph
                            di, dj = int(np.random.random() * 3) - 1, int(np.random.random() * 3) - 1
                    else:
                        # 3. Define a putative move
                        di, dj = int(np.random.random() * 3) - 1, int(np.random.random() * 3) - 1
                        while not (di + i >= 0) & (di + i < self.x.size) & (dj + j >= 0) & (dj + j < self.y.size) & (
                                (di != 0) or (dj != 0)):  # in some cases, re-sample if point chosen not within graph
                            di, dj = int(np.random.random() * 3) - 1, int(np.random.random() * 3) - 1
                    ii, jj = i + di, j + dj
                    dE = self.dEnergy(ID, i, j, di, dj)
                    if (dE < 0):
                        ID, xy_clls = self.swapper(ID, ID, i, j, ii, jj, xy_clls)
                    elif np.random.random() < np.exp(-dE / T):
                        ID, xy_clls = self.swapper(ID, ID, i, j, ii, jj, xy_clls)


                if (np.mod(t,check_freq)==0)or(subpopT+subpopE==3)or(t<check_freq): #check more regularly when near equilib
                    subpopE,subpopT,__ = self.find_subpopulations(self.get_C(ID))
                    self.ID = ID

                if t in self.t_save:
                        self.ID_save[np.where(t == self.t_save)[0][0]] = ID
                if t in self.t_print:
                    print("%d %% completed" % (100 * t / self.t_print[-1]))

                #
                # if ((np.mod(t,check_freq)==0)and(XEN_external_timescale==max_time))or(t<check_freq):
                #     if self.find_XEN_externalisation(self.get_C(ID))==1:
                #         XEN_external_timescale = t


                t = t+1

        self.t_equilib = t
        # self.XEN_external_timescale = XEN_external_timescale
        self.ID = ID
        # self.E = self.E_tot(C)



    def perform_simulation_dynamic(self,swap_rate=10,division=False):
        """Performs Metropolis-Hastings. Starting with C0, iterates a random cell selection and putative
       swapping procedure. When cells reach their division time, they undergo divisions

        swap_rate entails the number of cells (as a  proportion of the total number of cells of the embryoid as a whole)
            that are selected in each iteration.
            When swap_rate = 1, N cells are selected with each iteration, where N is the number of cells
            (Note that cell selection is still stochastic i.e. swap_rate =1 does NOT mean that every cell attempts
            to undergo a swap

        if division is False: only swapping is considered. swap_rate term is ignored, with one time-point
        considering one swap"""

        ID = self.ID0
        xy_clls = self.xy_clls
        N_cells = self.number_of_cells(self.C0)
        self.adjacency_time = self.get_adjacency(ID)
        # if division is True:
        #     Timer = self.Timer
        #     for t, T in enumerate(self.T_t):
        #
        #         #1. Divide
        #         Timer = Timer - 1
        #         if np.sum(Timer < 0)!=0:
        #             i_div, j_div = np.where(Timer<0)
        #             while i_div.size !=0:
        #                 id = np.random.randint(i_div.size)
        #                 i,j = i_div[id],j_div[id]
        #                 if self.boundary[i,j]:
        #                     poss_dirs = np.where(self.neighbourhood_possibility(self.well, i, j) == 1)[0]
        #                     dir = poss_dirs[int(poss_dirs.size*np.random.random())]
        #                 else:
        #                     dir = int(8*np.random.random())
        #                 ID,N_cells = self.divider(ID,i, j, dir,N_cells=N_cells)
        #                 Timer = self.divider(Timer,i, j, dir)
        #                 Timer[i,j] = np.random.normal(self.division_time,self.division_SD)
        #                 Timer[self.generate_iijj_from_dir(i,j,dir)] = np.random.normal(self.division_time,self.division_SD)
        #                 i_div, j_div = np.where(Timer < 0)
        #             x_clls, y_clls = np.where(ID != 0)
        #             xy_clls = np.array([x_clls, y_clls]).T
        #
        #
        #         for n in range(int(xy_clls.shape[0]/swap_rate)): #approximately scale the rate of swapping with embryo size, while allowing for division to occur psuedo-simultaneously.
        #             #2. Select a random cell
        #             cll_id = int(np.floor(np.random.random() * xy_clls.shape[0]))
        #             i, j = xy_clls[cll_id]
        #
        #             if self.boundary[i,j]:
        #                 # 3. Define a putative move
        #                 di, dj = self.ij_move()
        #                 while not (self.valid_move(i,j,di,dj)& self.boundary_valid_didj(self.well, i, j, di, dj)):  # in some cases, re-sample if point chosen not within graph
        #                     di, dj = self.ij_move()
        #             else:
        #                 #3. Define a putative move
        #                 di,dj = self.ij_move()
        #                 while not self.valid_move(i,j,di,dj):#in some cases, re-sample if point chosen not within graph
        #                     di, dj = self.ij_move()
        #             ii, jj = i+di, j+dj
        #             dE = self.dEnergy(ID,i,j,di,dj)
        #             if (dE < 0):
        #                 ID,xy_clls = self.swapper(ID,ID,i,j,ii,jj,xy_clls)
        #                 Timer = self.swapper(Timer,ID,i,j,ii,jj)
        #             elif np.random.random()< np.exp(-dE/T):
        #                 ID,xy_clls = self.swapper(ID,ID,i,j,ii,jj,xy_clls)
        #                 Timer = self.swapper(Timer,ID,i,j,ii,jj)
        #
        #             self.XXX = ID.copy()
        #
        #
        #         if t in self.t_save:
        #             self.ID_save[np.where(t==self.t_save)[0][0]] = ID
        #         if t in self.t_print:
        #             print("%d %% completed"%(100*t/self.t_print[-1]))
        #     self.Timer = Timer
        if division is False:
            for t, T in enumerate(self.T_t):
                self.adjacency_timer(ID)
                self.update_E_mat_dynamic()
                # 2. Select a random cell
                cll_id = int(np.floor(np.random.random() * xy_clls.shape[0]))
                i, j = xy_clls[cll_id]

                if self.boundary[i, j]:
                    # 3. Define a putative move
                    di, dj = self.ij_move()
                    while not (self.valid_move(i,j,di,dj)& self.boundary_valid_didj(self.well, i, j, di, dj)):  # in some cases, re-sample if point chosen not within graph
                        di, dj = self.ij_move()
                else:
                    # 3. Define a putative move
                    di, dj = self.ij_move()
                    while not self.valid_move(i,j,di,dj):  # in some cases, re-sample if point chosen not within graph
                        di, dj = self.ij_move()
                ii, jj = i + di, j + dj
                dE = self.dEnergy_dyn(ID, i, j, di, dj)
                if (dE < 0):
                    ID, xy_clls = self.swapper(ID, ID, i, j, ii, jj, xy_clls)
                elif np.random.random() < np.exp(-dE / T):
                    ID, xy_clls = self.swapper(ID, ID, i, j, ii, jj, xy_clls)

                if t in self.t_save:
                        self.ID_save[np.where(t == self.t_save)[0][0]] = ID
                if t in self.t_print:
                    print("%d %% completed" % (100 * t / self.t_print[-1]))
        self.ID = ID
        # self.E = self.E_tot(C)



    def perform_simulation_dynamicT(self,swap_rate=10,division=False):
        """Performs Metropolis-Hastings. Starting with C0, iterates a random cell selection and putative
       swapping procedure. When cells reach their division time, they undergo divisions

        swap_rate entails the number of cells (as a  proportion of the total number of cells of the embryoid as a whole)
            that are selected in each iteration.
            When swap_rate = 1, N cells are selected with each iteration, where N is the number of cells
            (Note that cell selection is still stochastic i.e. swap_rate =1 does NOT mean that every cell attempts
            to undergo a swap

        if division is False: only swapping is considered. swap_rate term is ignored, with one time-point
        considering one swap"""

        ID = self.ID0
        xy_clls = self.xy_clls
        N_cells = self.number_of_cells(self.C0)
        self.adjacency_time = self.get_adjacency(ID)
        # if division is True:
        #     Timer = self.Timer
        #     for t, T in enumerate(self.T_t):
        #
        #         #1. Divide
        #         Timer = Timer - 1
        #         if np.sum(Timer < 0)!=0:
        #             i_div, j_div = np.where(Timer<0)
        #             while i_div.size !=0:
        #                 id = np.random.randint(i_div.size)
        #                 i,j = i_div[id],j_div[id]
        #                 if self.boundary[i,j]:
        #                     poss_dirs = np.where(self.neighbourhood_possibility(self.well, i, j) == 1)[0]
        #                     dir = poss_dirs[int(poss_dirs.size*np.random.random())]
        #                 else:
        #                     dir = int(8*np.random.random())
        #                 ID,N_cells = self.divider(ID,i, j, dir,N_cells=N_cells)
        #                 Timer = self.divider(Timer,i, j, dir)
        #                 Timer[i,j] = np.random.normal(self.division_time,self.division_SD)
        #                 Timer[self.generate_iijj_from_dir(i,j,dir)] = np.random.normal(self.division_time,self.division_SD)
        #                 i_div, j_div = np.where(Timer < 0)
        #             x_clls, y_clls = np.where(ID != 0)
        #             xy_clls = np.array([x_clls, y_clls]).T
        #
        #
        #         for n in range(int(xy_clls.shape[0]/swap_rate)): #approximately scale the rate of swapping with embryo size, while allowing for division to occur psuedo-simultaneously.
        #             #2. Select a random cell
        #             cll_id = int(np.floor(np.random.random() * xy_clls.shape[0]))
        #             i, j = xy_clls[cll_id]
        #
        #             if self.boundary[i,j]:
        #                 # 3. Define a putative move
        #                 di, dj = self.ij_move()
        #                 while not (self.valid_move(i,j,di,dj)& self.boundary_valid_didj(self.well, i, j, di, dj)):  # in some cases, re-sample if point chosen not within graph
        #                     di, dj = self.ij_move()
        #             else:
        #                 #3. Define a putative move
        #                 di,dj = self.ij_move()
        #                 while not self.valid_move(i,j,di,dj):#in some cases, re-sample if point chosen not within graph
        #                     di, dj = self.ij_move()
        #             ii, jj = i+di, j+dj
        #             dE = self.dEnergy(ID,i,j,di,dj)
        #             if (dE < 0):
        #                 ID,xy_clls = self.swapper(ID,ID,i,j,ii,jj,xy_clls)
        #                 Timer = self.swapper(Timer,ID,i,j,ii,jj)
        #             elif np.random.random()< np.exp(-dE/T):
        #                 ID,xy_clls = self.swapper(ID,ID,i,j,ii,jj,xy_clls)
        #                 Timer = self.swapper(Timer,ID,i,j,ii,jj)
        #
        #             self.XXX = ID.copy()
        #
        #
        #         if t in self.t_save:
        #             self.ID_save[np.where(t==self.t_save)[0][0]] = ID
        #         if t in self.t_print:
        #             print("%d %% completed"%(100*t/self.t_print[-1]))
        #     self.Timer = Timer
        if division is False:
            for t, T in enumerate(self.T_t):
                self.adjacency_timer(ID)
                self.update_T_dynamic()
                # 2. Select a random cell
                cll_id = int(np.floor(np.random.random() * xy_clls.shape[0]))
                i, j = xy_clls[cll_id]

                if self.boundary[i, j]:
                    # 3. Define a putative move
                    di, dj = self.ij_move()
                    while not (self.valid_move(i,j,di,dj)& self.boundary_valid_didj(self.well, i, j, di, dj)):  # in some cases, re-sample if point chosen not within graph
                        di, dj = self.ij_move()
                else:
                    # 3. Define a putative move
                    di, dj = self.ij_move()
                    while not self.valid_move(i,j,di,dj):  # in some cases, re-sample if point chosen not within graph
                        di, dj = self.ij_move()
                ii, jj = i + di, j + dj
                dE = self.dEnergy(ID, i, j, di, dj)
                if (dE < 0):
                    ID, xy_clls = self.swapper(ID, ID, i, j, ii, jj, xy_clls)
                elif np.random.random() < np.exp(-dE / (T/(1+self.dynT[ID[i,j]]))):
                    ID, xy_clls = self.swapper(ID, ID, i, j, ii, jj, xy_clls)

                if t in self.t_save:
                        self.ID_save[np.where(t == self.t_save)[0][0]] = ID
                if t in self.t_print:
                    print("%d %% completed" % (100 * t / self.t_print[-1]))
        self.ID = ID
        # self.E = self.E_tot(C)





    def perform_simulation_to_equilibrium_dynamic_adhesion(self,swap_rate=10,division=False,max_time=1e6,check_freq=1e2,track_XEN=False,alpha=0.2):
        """Performs Metropolis-Hastings. Starting with C0, iterates a random cell selection and putative
       swapping procedure. When cells reach their division time, they undergo divisions

        swap_rate entails the number of cells (as a  proportion of the total number of cells of the embryoid as a whole)
            that are selected in each iteration.
            When swap_rate = 1, N cells are selected with each iteration, where N is the number of cells
            (Note that cell selection is still stochastic i.e. swap_rate =1 does NOT mean that every cell attempts
            to undergo a swap

        if division is False: only swapping is considered. swap_rate term is ignored, with one time-point
        considering one swap

        Performs until equilibrium is reached"""

        ID = self.ID0
        xy_clls = self.xy_clls
        N_cells = self.number_of_cells(self.C0)
        if division is True:
            print("Division not configured for this yet")
            # Timer = self.Timer
            # t = 0
            # T = self.T_t[0]
            # subpopE,subpopT = 10,10 #seed with a number > 1
            # while t < max_time:
            #     while (subpopE>1)&(subpopT>1):
            #         #1. Divide
            #         Timer = Timer - 1
            #         if np.sum(Timer < 0)!=0:
            #             i_div, j_div = np.where(Timer<0)
            #             while i_div.size !=0:
            #                 id = np.random.randint(i_div.size)
            #                 i,j = i_div[id],j_div[id]
            #                 if self.boundary[i,j]:
            #                     poss_dirs = np.where(self.neighbourhood_possibility(self.well, i, j) == 1)[0]
            #                     dir = poss_dirs[int(poss_dirs.size*np.random.random())]
            #                 else:
            #                     dir = int(8*np.random.random())
            #                 ID,N_cells = self.divider(ID,i, j, dir,N_cells=N_cells)
            #                 Timer = self.divider(Timer,i, j, dir)
            #                 Timer[i,j] = np.random.normal(self.division_time,self.division_SD)
            #                 Timer[self.generate_iijj_from_dir(i,j,dir)] = np.random.normal(self.division_time,self.division_SD)
            #                 i_div, j_div = np.where(Timer < 0)
            #             x_clls, y_clls = np.where(ID != 0)
            #             xy_clls = np.array([x_clls, y_clls]).T
            #
            #
            #         for n in range(int(xy_clls.shape[0]/swap_rate)): #approximately scale the rate of swapping with embryo size, while allowing for division to occur psuedo-simultaneously.
            #             #2. Select a random cell
            #             cll_id = int(np.floor(np.random.random() * xy_clls.shape[0]))
            #             i, j = xy_clls[cll_id]
            #
            #             if self.boundary[i,j]:
            #                 # 3. Define a putative move
            #                 di, dj = int(np.random.random() * 3) - 1, int(np.random.random() * 3) - 1
            #                 while not (di + i >= 0) & (di + i < self.x.size) & (dj + j >= 0) & (dj + j < self.y.size) & (
            #                         (di != 0) or (dj != 0))&self.boundary_valid_didj(self.well,i,j,di,dj):  # in some cases, re-sample if point chosen not within graph
            #                     di, dj = int(np.random.random() * 3) - 1, int(np.random.random() * 3) - 1
            #             else:
            #                 #3. Define a putative move
            #                 di,dj = int(np.random.random()*3) - 1, int(np.random.random()*3) - 1
            #                 while not (di+i>=0)&(di+i<self.x.size)&(dj+j>=0)&(dj+j<self.y.size)&((di!=0)or(dj!=0)):#in some cases, re-sample if point chosen not within graph
            #                     di, dj = int(np.random.random()*3) - 1,int(np.random.random()*3) - 1
            #             ii, jj = i+di, j+dj
            #             dE = self.dEnergy(ID,i,j,di,dj)
            #             if (dE < 0):
            #                 ID,xy_clls = self.swapper(ID,ID,i,j,ii,jj,xy_clls)
            #                 Timer = self.swapper(Timer,ID,i,j,ii,jj)
            #             elif np.random.random()< np.exp(-dE/T):
            #                 ID,xy_clls = self.swapper(ID,ID,i,j,ii,jj,xy_clls)
            #                 Timer = self.swapper(Timer,ID,i,j,ii,jj)
            #
            #
            #         if t in self.t_save:
            #             self.ID_save[np.where(t==self.t_save)[0][0]] = ID
            #         if t in self.t_print:
            #             print("%d %% completed"%(100*t/self.t_print[-1]))
            #         t +=1
            # self.Timer = Timer
        if division is False:
            t = 0
            subpopE,subpopT = 10,10 #seed with a number > 1
            XEN_external=0
            XEN_external_timescale=max_time
            while (t < max_time)and((subpopE>1)or(subpopT>1)):
                for n in range(self.N):
                    T = self.T_t[t]
                    # 2. Select a random cell
                    cll_id = int(np.floor(np.random.random() * xy_clls.shape[0]))
                    i, j = xy_clls[cll_id]

                    if self.boundary[i, j]:
                        # 3. Define a putative move
                        di, dj = int(np.random.random() * 3) - 1, int(np.random.random() * 3) - 1
                        while not (di + i >= 0) & (di + i < self.x.size) & (dj + j >= 0) & (dj + j < self.y.size) & (
                                (di != 0) or (dj != 0)) & self.boundary_valid_didj(self.well, i, j, di,
                                                                                   dj):  # in some cases, re-sample if point chosen not within graph
                            di, dj = int(np.random.random() * 3) - 1, int(np.random.random() * 3) - 1
                    else:
                        # 3. Define a putative move
                        di, dj = int(np.random.random() * 3) - 1, int(np.random.random() * 3) - 1
                        while not (di + i >= 0) & (di + i < self.x.size) & (dj + j >= 0) & (dj + j < self.y.size) & (
                                (di != 0) or (dj != 0)):  # in some cases, re-sample if point chosen not within graph
                            di, dj = int(np.random.random() * 3) - 1, int(np.random.random() * 3) - 1
                    ii, jj = i + di, j + dj
                    dE = self.dEnergy_dynamic(ID, i, j, di, dj,alpha)
                    if (dE < 0):
                        ID, xy_clls = self.swapper(ID, ID, i, j, ii, jj, xy_clls)
                    elif np.random.random() < np.exp(-dE / T):
                        ID, xy_clls = self.swapper(ID, ID, i, j, ii, jj, xy_clls)


                if (np.mod(t,check_freq)==0)or(subpopT+subpopE==3)or(t<check_freq): #check more regularly when near equilib
                    subpopE,subpopT,__ = self.find_subpopulations(self.get_C(ID))
                    self.ID = ID

                if t in self.t_save:
                        self.ID_save[np.where(t == self.t_save)[0][0]] = ID
                if t in self.t_print:
                    print("%d %% completed" % (100 * t / self.t_print[-1]))

                #
                # if ((np.mod(t,check_freq)==0)and(XEN_external_timescale==max_time))or(t<check_freq):
                #     if self.find_XEN_externalisation(self.get_C(ID))==1:
                #         XEN_external_timescale = t


                t = t+1

        self.t_equilib = t
        # self.XEN_external_timescale = XEN_external_timescale
        self.ID = ID
        # self.E = self.E_tot(C)


    def perform_simulation_dynamic_temperature(self,swap_rate=10,division=False,alpha = 1/300):
        """Performs Metropolis-Hastings. Starting with C0, iterates a random cell selection and putative
       swapping procedure. When cells reach their division time, they undergo divisions

        swap_rate entails the number of cells (as a  proportion of the total number of cells of the embryoid as a whole)
            that are selected in each iteration.
            When swap_rate = 1, N cells are selected with each iteration, where N is the number of cells
            (Note that cell selection is still stochastic i.e. swap_rate =1 does NOT mean that every cell attempts
            to undergo a swap

        if division is False: only swapping is considered. swap_rate term is ignored, with one time-point
        considering one swap"""

        ID = self.ID0
        C = self.C0
        xy_clls = self.xy_clls
        N_cells = self.number_of_cells(self.C0)
        V = np.zeros_like(C)
        if division is True:
            Timer = self.Timer
            for t, T in enumerate(self.T_t):

                #1. Divide
                Timer = Timer - 1
                if np.sum(Timer < 0)!=0:
                    i_div, j_div = np.where(Timer<0)
                    while i_div.size !=0:
                        id = np.random.randint(i_div.size)
                        i,j = i_div[id],j_div[id]
                        if self.boundary[i,j]:
                            poss_dirs = np.where(self.neighbourhood_possibility(self.well, i, j) == 1)[0]
                            dir = poss_dirs[int(poss_dirs.size*np.random.random())]
                        else:
                            dir = int(8*np.random.random())
                        ID,N_cells = self.divider(ID,i, j, dir,N_cells=N_cells)
                        C = self.divider(C,i, j, dir)
                        Timer = self.divider(Timer,i, j, dir)
                        Timer[i,j] = np.random.normal(self.division_time,self.division_SD)
                        Timer[self.generate_iijj_from_dir(i,j,dir)] = np.random.normal(self.division_time,self.division_SD)
                        i_div, j_div = np.where(Timer < 0)
                    x_clls, y_clls = np.where(ID != 0)
                    xy_clls = np.array([x_clls, y_clls]).T


                for n in range(int(xy_clls.shape[0]/swap_rate)): #approximately scale the rate of swapping with embryo size, while allowing for division to occur psuedo-simultaneously.
                    # V = self.contact_integration(C,V,k_gain=0.1,k_loss=0.001)*(C!=0)
                    #2. Select a random cell
                    cll_id = int(np.floor(np.random.random() * xy_clls.shape[0]))
                    i, j = xy_clls[cll_id]

                    if self.boundary[i,j]:
                        # 3. Define a putative move
                        di, dj = int(np.random.random() * 3) - 1, int(np.random.random() * 3) - 1
                        while not (di + i >= 0) & (di + i < self.x.size) & (dj + j >= 0) & (dj + j < self.y.size) & (
                                (di != 0) or (dj != 0))&self.boundary_valid_didj(self.well,i,j,di,dj):  # in some cases, re-sample if point chosen not within graph
                            di, dj = int(np.random.random() * 3) - 1, int(np.random.random() * 3) - 1
                    else:
                        #3. Define a putative move
                        di,dj = int(np.random.random()*3) - 1, int(np.random.random()*3) - 1
                        while not (di+i>=0)&(di+i<self.x.size)&(dj+j>=0)&(dj+j<self.y.size)&((di!=0)or(dj!=0)):#in some cases, re-sample if point chosen not within graph
                            di, dj = int(np.random.random()*3) - 1,int(np.random.random()*3) - 1
                    ii, jj = i+di, j+dj
                    dE = self.dEnergy(ID,i,j,di,dj)
                    if (dE < 0):
                        ID,xy_clls = self.swapper(ID,ID,i,j,ii,jj,xy_clls)
                        Timer = self.swapper(Timer,ID,i,j,ii,jj)
                        C = self.swapper(C,ID,i,j,ii,jj)
                    elif np.random.random()< np.exp(-dE*self.self_contact_ij(C,i,j)**2/T): #
                        ID,xy_clls = self.swapper(ID,ID,i,j,ii,jj,xy_clls)
                        Timer = self.swapper(Timer,ID,i,j,ii,jj)
                        C = self.swapper(C,ID,i,j,ii,jj)



                if t in self.t_save:
                    self.ID_save[np.where(t==self.t_save)[0][0]] = ID
                if t in self.t_print:
                    print("%d %% completed"%(100*t/self.t_print[-1]))
            self.Timer = Timer
        if division is False:
            for t, T in enumerate(self.T_t):
                # 2. Select a random cell
                cll_id = int(np.floor(np.random.random() * xy_clls.shape[0]))
                i, j = xy_clls[cll_id]

                if self.boundary[i, j]:
                    # 3. Define a putative move
                    di, dj = int(np.random.random() * 3) - 1, int(np.random.random() * 3) - 1
                    while not (di + i >= 0) & (di + i < self.x.size) & (dj + j >= 0) & (dj + j < self.y.size) & (
                            (di != 0) or (dj != 0)) & self.boundary_valid_didj(self.well, i, j, di,
                                                                               dj):  # in some cases, re-sample if point chosen not within graph
                        di, dj = int(np.random.random() * 3) - 1, int(np.random.random() * 3) - 1
                else:
                    # 3. Define a putative move
                    di, dj = int(np.random.random() * 3) - 1, int(np.random.random() * 3) - 1
                    while not (di + i >= 0) & (di + i < self.x.size) & (dj + j >= 0) & (dj + j < self.y.size) & (
                            (di != 0) or (dj != 0)):  # in some cases, re-sample if point chosen not within graph
                        di, dj = int(np.random.random() * 3) - 1, int(np.random.random() * 3) - 1
                ii, jj = i + di, j + dj
                dE = self.dEnergy(ID, i, j, di, dj)
                if (dE < 0):
                    ID, xy_clls = self.swapper(ID, ID, i, j, ii, jj, xy_clls)
                    C = self.swapper(C, C, i, j, ii, jj)

                elif np.random.random() < np.exp(-dE *np.exp(self.self_contact_ij(C,i,j)-(8/3))/ T):
                    ID, xy_clls = self.swapper(ID, ID, i, j, ii, jj, xy_clls)
                    C = self.swapper(C, C, i, j, ii, jj)

                if t in self.t_save:
                        self.ID_save[np.where(t == self.t_save)[0][0]] = ID
                if t in self.t_print:
                    print("%d %% completed" % (100 * t / self.t_print[-1]))
        self.ID = ID
        self.V = V
        # self.E = self.E_tot(C)



    def get_C(self,ID):
        """Returns C matrix from the ID matrix"""
        C = np.mod(ID, 3)
        C[C == 0] = 3
        C = C * (ID != 0)
        return C

    def get_C_save(self):
        """Returns C matrix from ID matrix for all t"""
        C_save = np.zeros_like(self.ID_save)
        for i, ID in enumerate(self.ID_save):
            C_save[i] = self.get_C(ID)
        self.C_save = C_save
        return C_save

    def find_energies(self):
        """Calculates the global energy for each time-point in t_save, """
        E_t = np.array([self.E_tot(self.ID_save[i].astype(int)) for i in range(self.n_save)])
        self.E_t = E_t
        return E_t

    def find_cell_numbers(self):
        """Calculates cell number for each cell type for each time point in t_save"""
        cell_number = np.zeros(self.C_save.shape[2])
        for i in range(self.C_save.shape[2]):
            cell_number[i] = np.sum(self.C_save[:,:,i]!=0)
        self.cell_number_t = cell_number
        return cell_number

    def find_subpopulations(self,C):
        """For each cell type, finds the number of subpopulations (i.e. cases where cells or clusters of cells are
        seperated by more than one element-width (including diagonals).

        Returns a list: ES,TS,XEN"""
        clique_sizes = []
        for i in np.arange(1,4):
            grid = 1 * (C == i)
            g = Graph()
            clique_sizes.append(g.numIslandsDFS(grid.tolist()))
        self.subpopulation_number = clique_sizes
        return clique_sizes

    def find_subpopulations_t(self):
        """Finds subpopulations for all t in t_save"""
        clique_size_t = np.array([np.array(self.find_subpopulations(self.C_save[i])) for i in range(self.n_save)])
        self.subpopulation_number_t = clique_size_t
        return clique_size_t

    def find_average_self_self_contacts(self,C):
        """Calculates the average number of self-self contacts per cell. Gives an indication of flattness vs roundness
        of a colony or of colonies if there are multiple"""
        av_ss_c = []
        for i in np.arange(1,4):
            x_clls, y_clls = np.where(C==i)
            sum_contacts = 0
            for j, x in enumerate(x_clls):
                y = y_clls[j]
                sum_contacts += np.sum(C[x-1:x+2,y-1:y+2]==i)-1
            av_ss_c.append(sum_contacts/x_clls.size)
        self.average_self_self = av_ss_c
        return av_ss_c

    def find_average_self_self_contacts_t(self):
        """Finds average self-self contacts for all t in t_save"""
        av_ss_c_t = np.array([np.array(self.find_average_self_self_contacts(self.C_save[i])) for i in range(self.n_save)])
        self.average_self_self_t = av_ss_c_t

    def fit_ellipse(self,C):
        """Fits an ellipse to non-zero values in the configuration matrix C

        x0,y0 are the centre of the ellipse

        xl,yl are the short and long axis lengths

        theta is the angle of rotation"""
        img = np.uint8(255*(C!=0))
        cnts, hiers = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        cntss = []
        for cnt in cnts:
            if len(cnt)>5:
                cntss.append(cnt)
        if len(cntss)>1:
            xls = np.zeros(len(cntss))
            yls = np.zeros(len(cntss))
            for cnt in cntss:
                (x0,y0),(xl,yl),theta = cv2.fitEllipse(cnt)
                xls[xl]
                yls[yl]
            cnt = cntss[np.where(xls*yls==np.max(xls*yls))]
        else:
            cnt = cntss[0]
        (x0, y0), (xl, yl), theta = cv2.fitEllipse(cnt)

        return (x0, y0), (xl, yl), theta

    def find_eccentricity(self,C):
        """Finds the eccentricity of a cluster of cells. This performs ellipse fitting."""
        (x0, y0), __, __ = self.fit_ellipse(C)
        if x0>y0:
            a,b = x0,y0
        else:
            a,b = y0,x0
        ecc = np.sqrt(1 - b ** 2 / a ** 2)
        self.eccentricity = ecc
        return ecc

    def find_eccentricity_t(self):
        """Finds eccentricities of clusters for all t_save"""
        ecc_t = np.array([self.find_eccentricity(C) for C in self.C_save])
        self.eccentricity_t = ecc_t
        return ecc_t

    def find_angular_distribution(self,C):
        """Defines the angular distribution of cells across the embryoid.

        Fits an ellipse to find the centroid. Then considers half-lines from this centroid over a 360deg rotation.
        Tracks the ***proportion*** of cells of each type.

        Should give an indication of polarity: if cells are radially symmetric, the angular distribution is flat. e.g. XEN
        If they are segregated to one side, then expect a sinosoidal wave e.g. TS/ES

        BETA: CHECK!!"""
        (x0, y0), __, __ = self.fit_ellipse(C)
        theta_space = np.linspace(0,2*np.pi,50)
        dtheta = theta_space[1] - theta_space[0]
        r_max = np.min([int(C.shape[0]-x0),int(x0),int(C.shape[0]-y0),int(y0)])
        r_space = np.linspace(0,r_max,100)
        p_E,p_T,p_X = np.zeros(theta_space.size),np.zeros(theta_space.size),np.zeros(theta_space.size)
        for i, theta in enumerate(theta_space):
            theta_sample_n = 10
            x = x0 + np.outer(r_space,np.cos(np.linspace(theta,theta+dtheta,theta_sample_n)))
            y = y0 + np.outer(r_space,np.sin(np.linspace(theta,theta+dtheta,theta_sample_n)))
            xy = np.array([np.round(x.flatten()),np.round(y.flatten())]).astype(int)
            xy = np.unique(xy,axis=0)
            sample = C[xy[0],xy[1]]
            p_E[i] = np.sum(sample==1)/np.sum(sample!=0)
            p_T[i] = np.sum(sample == 2) / np.sum(sample != 0)
            p_X[i] = np.sum(sample == 3) / np.sum(sample != 0)
        self.angular_distribution = np.array([p_E,p_T,p_X])
        return p_E,p_T,p_X

    def find_angular_polarity(self,C):
        """
        Quantifies the angular polarity by finding the angular distribution and determining the
        amplitude of a fitted sinosoidal wave (with period 360deg/2pi).

        Complete polarisation should entail a polarity of 0.5
        Unpolarised entails a polarity of 0
        """
        p_E, p_T, p_X = self.find_angular_distribution(C)

        def fit_sine(X, p):
            """
            Cost function to fit a sinosoidal wave with period 2pi
            X = [y-offset,amplitude,x-offset
            """
            sine = X[0] + X[1] * np.sin(np.pi * 2 * np.arange(p.size) / p.size - X[2])
            return np.sum(np.sqrt((p - sine) ** 2))

        amps = []
        for p in [p_E,p_T,p_X]:
            res = minimize(fit_sine,[0.5,0.5,0],method="Powell",args=(p,))
            amps.append(np.absolute(res.x[1]))
        amps = np.array(amps)
        self.angular_polarity = amps
        return amps

    def find_angular_polarity_t(self):
        """Finds angular polarity for t in t_save"""
        a_pol_t = np.array([np.array(self.find_angular_polarity(self.C_save[i])) for i in range(self.n_save)])
        self.angular_polarity_t = a_pol_t
        return a_pol_t

    def find_XEN_externalisation(self,C,membrane_contacts=1,any_contacts=3):
        """Finds proportion of XEN cells on outside"""
        XEN = 1*(C==3)
        number_of_membrane_contacts = 8*XEN - XEN*(np.roll(XEN,-1,axis=0)+np.roll(XEN,1,axis=0)+np.roll(XEN,-1,axis=1)+np.roll(XEN,1,axis=1))
        ANY = 1 * (C !=0)
        number_of_any_contacts = 8*XEN - XEN*(np.roll(ANY,-1,axis=0)+np.roll(ANY,1,axis=0)+np.roll(ANY,-1,axis=1)+np.roll(ANY,1,axis=1))
        return np.sum((number_of_membrane_contacts>=membrane_contacts)+(number_of_any_contacts>=any_contacts))/np.sum(XEN)

    def find_XEN_externalisation_t(self):
        """FInds proprotion of XEN on outside for all t"""
        return np.array([self.find_XEN_externalisation(C) for C in self.C_save])

    def cluster_index(self,C,i_s = (1,2),radial=False,d_eq = False):
        num_x, num_y = C.shape
        X, Y = np.meshgrid(np.arange(num_x), np.arange(num_y), indexing="ij")
        Dd = cdist(np.array([X.ravel(), Y.ravel()]).T, np.array([X.ravel(), Y.ravel()]).T, metric='chebyshev')
        def get_p(i,d):
            CC = 1.0*(C==i)
            CCC = np.outer(CC,CC)
            NN = 1.0*(C!=0)
            NNN = np.outer(NN,NN)
            if d_eq is False:
                p = np.sum(CCC[Dd<=d])/np.sum(NNN[Dd<=d])
                return p
            else:
                p = np.sum(CCC[Dd == d]) / np.sum(NNN[Dd == d])
                return p
        if radial is False:
            self._cluster_index = [get_p(i, Dd.max()) for  i in i_s]
            return self._cluster_index
        else:
            D_r = [np.array([get_p(i,d) for d in np.unique(Dd)]) for i in i_s]
            self._cluster_index_r = D_r
            return D_r


    def cluster_index_t(self,i_s=(1,2)):
        D_t = np.array([self.cluster(C,i_s,radial=False) for C in self.C_save])
        self._cluster_index_t = D_t
        return D_t

    def get_clustered(self,Ci):
        g = Graph()
        count, grid = g.assign_islands(Ci.tolist())
        return (-np.array(grid)).astype(int)

    def cluster_index2(self, C, i_s=(1, 2), radial=False):
        num_x, num_y = C.shape
        X, Y = np.meshgrid(np.arange(num_x), np.arange(num_y), indexing="ij")
        Dd = cdist(np.array([X.ravel(), Y.ravel()]).T, np.array([X.ravel(), Y.ravel()]).T, metric='chebyshev')
        clustered = [self.get_clustered(C==i) for i in i_s]
        def get_p(ii, d):
            c = clustered[ii]
            ni = np.unique(c)
            ni = ni[ni!=0]
            ns = np.zeros_like(ni)
            k = 0
            P = 0
            for j in ni:
                nn = np.sum(c == j).astype(int)
                ns[k] = nn
                CC = 1.0 * (c == j)
                CCC = np.outer(CC, CC)
                NN = 1.0 * (C != 0)
                NNN = np.outer(NN, NN)
                # p = np.sum(CCC[Dd <= d]) / np.sum(NNN[Dd <= d])
                P += np.sum(CCC[Dd <= d]) / np.sum(NNN[Dd <= d])*nn
                k+=1
            return P/np.sum(ns)
        if radial is False:
            self._cluster_index = [get_p(i, Dd.max()) for i in range(len(i_s))]
            return self._cluster_index
        else:
            D_r = [np.array([get_p(i, d) for d in np.unique(Dd)]) for i in range(len(i_s))]
            self._cluster_index_r = D_r
            return D_r

    def cluster_index_t2(self,i_s=(1,2),radial=False):
        D_t = np.array([self.cluster_index2(C,i_s,radial=radial) for C in self.C_save])
        self._cluster_index_t = D_t
        return D_t

    def circles(self,x, y, s, c='b', ax = None,vmin=None, vmax=None, **kwargs):
        """
        Make a scatter of circles plot of x vs y, where x and y are sequence
        like objects of the same lengths. The size of circles are in data scale.

        Parameters
        ----------
        x,y : scalar or array_like, shape (n, )
            Input data
        s : scalar or array_like, shape (n, )
            Radius of circle in data unit.
        c : color or sequence of color, optional, default : 'b'
            `c` can be a single color format string, or a sequence of color
            specifications of length `N`, or a sequence of `N` numbers to be
            mapped to colors using the `cmap` and `norm` specified via kwargs.
            Note that `c` should not be a single numeric RGB or RGBA sequence
            because that is indistinguishable from an array of values
            to be colormapped. (If you insist, use `color` instead.)
            `c` can be a 2-D array in which the rows are RGB or RGBA, however.
        vmin, vmax : scalar, optional, default: None
            `vmin` and `vmax` are used in conjunction with `norm` to normalize
            luminance data.  If either are `None`, the min and max of the
            color array is used.
        kwargs : `~matplotlib.collections.Collection` properties
            Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls),
            norm, cmap, transform, etc.

        Returns
        -------
        paths : `~matplotlib.collections.PathCollection`

        Examples
        --------
        a = np.arange(11)
        circles(a, a, a*0.2, c=a, alpha=0.5, edgecolor='none')
        plt.colorbar()

        License
        --------
        This code is under [The BSD 3-Clause License]
        (http://opensource.org/licenses/BSD-3-Clause)
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        from matplotlib.collections import PatchCollection

        if np.isscalar(c):
            kwargs.setdefault('color', c)
            c = None
        if 'fc' in kwargs: kwargs.setdefault('facecolor', kwargs.pop('fc'))
        if 'ec' in kwargs: kwargs.setdefault('edgecolor', kwargs.pop('ec'))
        if 'ls' in kwargs: kwargs.setdefault('linestyle', kwargs.pop('ls'))
        if 'lw' in kwargs: kwargs.setdefault('linewidth', kwargs.pop('lw'))

        patches = [Circle((x_, y_), s_) for x_, y_, s_ in np.broadcast(x, y, s)]
        collection = PatchCollection(patches, **kwargs)
        if c is not None:
            collection.set_array(np.asarray(c))
            collection.set_clim(vmin, vmax)
        if ax is None:
            ax = plt.gca()
        ax.add_collection(collection)
        ax.autoscale_view()
        if c is not None:
            plt.sci(collection)
        return collection

    def plot_cells(self,ax,C,id,col,**kwargs):
        """Plots cells of a given id with a specific colour

        ax = the axis on which cells are plotted

        C = configuration matrix

        id = cell id to be plotted

        col = colour"""
        x,y = np.where(C==id)
        self.circles(x,y,s=0.5,ax=ax,color=col,**kwargs)

    def plot_all(self,ax,C,cols=("red","blue","green"),**kwargs):
        """Plots the ETX onto ax"""
        E_col,T_col,X_col = cols
        self.plot_cells(ax,C,1,E_col,label="ES",**kwargs)
        self.plot_cells(ax,C,2,T_col,label="TS",**kwargs)
        self.plot_cells(ax,C,3,X_col,label="XEN",**kwargs)


    def plot_save(self,C,file_name=None,dir_name="plots",xlim=None,ylim=None,**kwargs):
        """Plots the ETX embryoid

        dir_name is the directory name

        file_name is a custom filename. If none is given, then the filename is time-stamped

        xlim,ylim define the size of the box. Will auto-scale if not assigned"""
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        fig, ax = plt.subplots()
        self.plot_all(ax,C,**kwargs)
        ax.axis('off')

        if xlim is None or ylim is None:
            x, y = np.where(C != 0)
            xlim, ylim = (x.min() - 1, x.max() + 1), (y.min() - 1, y.max() + 1)
        ax.set(aspect=1,xlim=xlim,ylim=ylim)

        if file_name is None:
            file_name = "embryoid%d"%time.time()

        fig.savefig("%s/%s.pdf"%(dir_name,file_name))

    def plot_time_series(self,n=6,xlim=None,ylim=None,file_name=None,dir_name="plots",**kwargs):
        """Plots a time-series of size n.

        xlim,ylim is the box size. Will auto-scale (keeping proportions across plots) if None given"""
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        fig, ax = plt.subplots(1,n,figsize=(n*2,2))
        if xlim is None or ylim is None:
            t,x,y = np.where(self.C_save!=0)
            xlim,ylim= (x.min()-1,x.max()+1),(y.min()-1,y.max()+1)
        for nn,i in enumerate(np.linspace(0,self.C_save.shape[2]-1,n).astype(int)):
            CC = self.C_save[i]
            self.plot_all(ax[nn],CC,**kwargs)
            ax[nn].axis('off')
            ax[nn].set(aspect=1,xlim=xlim,ylim=ylim)
        if file_name is None:
            file_name = "time_series %d"%time.time()
        fig.savefig("%s/%s.pdf"%(dir_name,file_name))


    def animate_C(self,file_name=None,dir_name="plots",xlim=None,ylim=None,plot_boundary=False,**kwargs):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        if xlim is None or ylim is None:
            t,x,y = np.where(self.C_save!=0)
            xlim,ylim= (x.min()-1,x.max()+1),(y.min()-1,y.max()+1)

        if plot_boundary is True:
            xlim,ylim=(0,self.x.max()),(0,self.y.max())
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)

        def animate(i):
            ax1.clear()
            ax1.set(aspect=1, xlim=xlim, ylim=ylim)
            ax1.axis('off')
            self.plot_all(ax1,self.C_save[i],**kwargs)
            if plot_boundary is True:
                ax1.imshow(self.well,cmap=cm.Greys)

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=20, bitrate=1800)

        if file_name is None:
            file_name = "animation %d"%time.time()

        an = animation.FuncAnimation(fig, animate, frames=self.n_save,interval=200)
        an.save("%s/%s.mp4"%(dir_name,file_name), writer=writer,dpi=264)

class ETX_continuous:
    def __init__(self):
        self.x_size, self.y_size = [],[]
        self.X = []
        self.X0 = []
        self.N_E, self.N_X, self.N_T, self.N = [], [], [], []
        self.system_size = []
        self.colors = ["red","blue","green"]

        self.W = []
        self.sigma0 = []
        self.E_mat = []
        self.E_mat_condensed = []

        self.tau = 0.005#0.005

    def set_domain(self,x_size,y_size):
        self.x_size,self.y_size = x_size,y_size
        self.xy_size = np.array([self.x_size,self.y_size])

    def set_particles(self,N_E,N_T,N_X,scaler):
        self.N_E,self.N_T,self.N_X = int(N_E*scaler),int(N_T*scaler),int(N_X*scaler)
        self.N = self.N_E + self.N_X + self.N_T

    def define_interaction_energies(self,W0,sigma0 = None):
        self.W = W0
        if sigma0 is None:
            sigma0 = np.zeros_like(self.W)
        self.sigma0 = sigma0
        E_mat = np.zeros([self.N,self.N])
        N_list = [0,self.N_E,self.N_T,self.N_X]
        N_cum_list = np.cumsum(N_list)
        self.dictionary = np.zeros(self.N)
        for i in range(3):
            self.dictionary[N_cum_list[i]:N_cum_list[i+1]] = i
        self.ID_mat_i,self.ID_mat_j = np.meshgrid(self.dictionary,self.dictionary,indexing="ij")
        for i in range(3):
            for j in range(3):
                E_mat[(self.ID_mat_i==i)&(self.ID_mat_j==j)] = -np.random.normal(self.W[i, j], self.sigma0[i, j],(N_list[i+1]*N_list[j+1]))
        mask = np.zeros_like(E_mat)
        rng = np.arange(E_mat.shape[0])
        for i in rng:
            mask[rng[:-i], rng[:-i]+i] = 1
        E_mat = E_mat*mask
        E_mat = E_mat + E_mat.T
        self.E_mat = E_mat
        self.E_mat_condensed = squareform(E_mat)



    def make_X0(self):
        X = np.array([np.random.uniform(self.x_size/4,3*self.x_size/4,self.N),np.random.uniform(self.y_size/4,3*self.y_size/4,self.N)])
        self.X0 = X
        self.X = X


    def circles(self,x, y, s, c='b', ax = None,vmin=None, vmax=None, **kwargs):
        """
        Make a scatter of circles plot of x vs y, where x and y are sequence
        like objects of the same lengths. The size of circles are in data scale.

        Parameters
        ----------
        x,y : scalar or array_like, shape (n, )
            Input data
        s : scalar or array_like, shape (n, )
            Radius of circle in data unit.
        c : color or sequence of color, optional, default : 'b'
            `c` can be a single color format string, or a sequence of color
            specifications of length `N`, or a sequence of `N` numbers to be
            mapped to colors using the `cmap` and `norm` specified via kwargs.
            Note that `c` should not be a single numeric RGB or RGBA sequence
            because that is indistinguishable from an array of values
            to be colormapped. (If you insist, use `color` instead.)
            `c` can be a 2-D array in which the rows are RGB or RGBA, however.
        vmin, vmax : scalar, optional, default: None
            `vmin` and `vmax` are used in conjunction with `norm` to normalize
            luminance data.  If either are `None`, the min and max of the
            color array is used.
        kwargs : `~matplotlib.collections.Collection` properties
            Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls),
            norm, cmap, transform, etc.

        Returns
        -------
        paths : `~matplotlib.collections.PathCollection`

        Examples
        --------
        a = np.arange(11)
        circles(a, a, a*0.2, c=a, alpha=0.5, edgecolor='none')
        plt.colorbar()

        License
        --------
        This code is under [The BSD 3-Clause License]
        (http://opensource.org/licenses/BSD-3-Clause)
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        from matplotlib.collections import PatchCollection

        if np.isscalar(c):
            kwargs.setdefault('color', c)
            c = None
        if 'fc' in kwargs: kwargs.setdefault('facecolor', kwargs.pop('fc'))
        if 'ec' in kwargs: kwargs.setdefault('edgecolor', kwargs.pop('ec'))
        if 'ls' in kwargs: kwargs.setdefault('linestyle', kwargs.pop('ls'))
        if 'lw' in kwargs: kwargs.setdefault('linewidth', kwargs.pop('lw'))

        patches = [Circle((x_, y_), s_) for x_, y_, s_ in np.broadcast(x, y, s)]
        collection = PatchCollection(patches, **kwargs)
        if c is not None:
            collection.set_array(np.asarray(c))
            collection.set_clim(vmin, vmax)
        if ax is None:
            ax = plt.gca()
        ax.add_collection(collection)
        ax.autoscale_view()
        if c is not None:
            plt.sci(collection)
        return collection

    def plot_all(self,ax,X):
        N_list = [0,self.N_E,self.N_T,self.N_X]
        N_cum_list = np.cumsum(N_list)
        for i in range(3):
            # ax.scatter(X[0,N_cum_list[i]:N_cum_list[i]+N_cum_list[i+1]],X[1,N_cum_list[i]:N_cum_list[i]+N_cum_list[i+1]],color=self.colors[i])
            self.circles(X[0,N_cum_list[i]:N_cum_list[i]+N_cum_list[i+1]],X[1,N_cum_list[i]:N_cum_list[i]+N_cum_list[i+1]],s=self.rm/2,color=self.colors[i])
    def plot_show(self,X):
        fig, ax = plt.subplots()
        self.plot_all(ax,X)
        ax.set(aspect=1)
        fig.show()

    def potential(self,r,E,type="Lennard-Jones"):
        rm = 0.5
        self.rm = rm
        # return -E*((rm/r)**12 - 2*(rm/r)**6)
        # return -E*((rm/r)**12 - 2*(rm/r)**6)*(r < self.rm*1.2)
        return -E*((rm/r)**30 - 2*(rm/r)**6)*(r < self.rm*2)


    def E_tot(self,X):
        return np.sum(self.potential(pdist(X.T),self.E_mat_condensed))

    def perform_simualtion(self,T_t = np.repeat(1,10**4)):
        X = self.X
        t_save = np.linspace(0,T_t.size-1,100).astype(int)
        self.t_save = t_save
        self.X_save = np.zeros([100,X.shape[0],X.shape[1]])
        ts = 0
        E_tot = self.E_tot(X)
        for t, T in enumerate(T_t):

            E_tot = self.E_tot(X)
            i = int(np.random.random()*self.N)
            move_x,move_y = self.tau*np.random.normal(0,1,2)
            X_new = X.copy()
            new_pos = X_new[:,i] + np.array([move_x,move_y])
            # while (new_pos[0]>=0)&(new_pos[0]<=self.x_size)&(new_pos[1]>=0)&(new_pos[1]<=self.y_size):
            #     move_x, move_y = self.tau * np.random.normal(0, 1, 2)
            #     new_pos += np.array([move_x, move_y])
            #     print("bounced")
            X_new[:,i] = (self.xy_size-np.absolute(np.absolute(new_pos)-self.xy_size)) #hard square boundary
            # X_new[:,i] = new_pos
            E_tot_new = self.E_tot(X_new)
            dE = E_tot_new - E_tot

            if dE < 0:
                X = X_new
                E_tot = E_tot_new
            if dE >= 0:
                if np.random.random()<np.exp(-dE/T):
                    X = X_new
                    E_tot = E_tot_new
            if t in t_save:
                self.X_save[ts]  = X
                ts+=1
        self.X = X
        #Could speed up by considering "neighbourhood" matrix and updating this as applicable.
        # Then can treat changes in LG potential as only considering these vals

    #
    # def perform_simualtion_pairs(self,T_t = np.repeat(1,10**4)):
    #     X = self.X
    #     E_tot = self.E_tot(X)
    #     for t, T in enumerate(T_t):
    #         i = int(np.random.random()*self.N)
    #         move_x,move_y = self.tau*np.random.normal(0,1,2)
    #         X_new = X.copy()
    #         new_pos = X_new[:,i] + np.array([move_x,move_y])
    #         # while (new_pos[0]>=0)&(new_pos[0]<=self.x_size)&(new_pos[1]>=0)&(new_pos[1]<=self.y_size):
    #         #     move_x, move_y = self.tau * np.random.normal(0, 1, 2)
    #         #     new_pos += np.array([move_x, move_y])
    #         #     print("bounced")
    #         X_new[:,i] = new_pos
    #
    #         neighbours = np.where(np.linalg.norm(X[:,i].T-X.T,axis=0)<self.rm)[0]
    #         if neighbours.size!=0:
    #             j = neighbours[int(np.random.random()*neighbours.size)]
    #             move_x, move_y = self.tau * np.random.normal(0, 1, 2)
    #             X_new = X.copy()
    #             new_pos = X_new[:, j] + np.array([move_x, move_y])
    #             X_new[:, j] = new_pos
    #
    #         E_tot_new = self.E_tot(X_new)
    #         dE = E_tot_new - E_tot
    #
    #         if dE < 0:
    #             X = X_new
    #             E_tot = E_tot_new
    #         if dE > 0:
    #             if np.random.random()<np.exp(-dE/T):
    #                 X = X_new
    #                 E_tot = E_tot_new
    #     self.X = X


    def animate_X(self,file_name=None,dir_name="plots",xlim=None,ylim=None,plot_boundary=False,**kwargs):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        if xlim is None or ylim is None:
            x,y = self.X_save[:,0,:].flatten(),self.X_save[:,1,:].flatten()
            # xlim,ylim= (x.min()-1,x.max()+1),(y.min()-1,y.max()+1)
            xlim,ylim= (x.min()-self.rm,x.max()+self.rm),(y.min()-self.rm,y.max()+self.rm)


        if plot_boundary is True:
            xlim,ylim=(0,self.x.max()),(0,self.y.max())
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)

        def animate(i):
            ax1.clear()
            ax1.set(aspect=1, xlim=xlim, ylim=ylim)
            ax1.axis('off')
            self.plot_all(ax1,self.X_save[i],**kwargs)

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=20, bitrate=1800)

        if file_name is None:
            file_name = "animation %d"%time.time()

        an = animation.FuncAnimation(fig, animate, frames=np.arange(self.t_save.size),interval=200)
        an.save("%s/%s.mp4"%(dir_name,file_name), writer=writer,dpi=264)


@jit(nopython=True,cache=True)
def dEnergy(E_mat,ID, i, j, di, dj):
    """Defines the energy change associated with a swap.
    i,j define the matrix position of the element in question
    di,dj define the direction of the putatively swapped cell (di,dj = -1,0,1)

    NB: verbose code, but optimised for efficiency"""
    dE = 0
    II,JJ = ID[i,j], ID[i+di,j+dj]
    if di == 0:
        for I in range(i - 1, i + 2):
            dE -= E_mat[JJ,ID[I, j + 2 * dj]]
            dE -= E_mat[II, ID[I, j -dj]]
            dE += E_mat[II, ID[I, j + 2 * dj]]
            dE += E_mat[JJ, ID[I, j - dj]]

    elif dj == 0:
        for J in range(j - 1, j + 2):
            dE -= E_mat[JJ, ID[i + 2 * di, J]]
            dE -= E_mat[II, ID[i - di, J]]
            dE += E_mat[II, ID[i + 2 * di, J]]
            dE += E_mat[JJ, ID[i - di, J]]
    else:
        ID_flat1 = ID[i - 1:i + 2, j - 1:j + 2].ravel()
        ID_flat2 = ID[i + di - 1:i + di + 2, j + dj - 1:j + dj + 2].ravel()
        if (di == 1) & (dj == 1):
            ids1 = ID_flat1.take([0, 1, 2, 3, 6])
            ids2 = ID_flat2.take([2, 5, 6, 7, 8])

        if (di == -1) & (dj == -1):
            ids1 = ID_flat1.take([2, 5, 6, 7, 8])
            ids2 = ID_flat2.take([0, 1, 2, 3, 6])

        if (di == 1) & (dj == -1):
            ids1 = ID_flat1.take([0, 1, 2, 5, 8])
            ids2 = ID_flat2.take([0, 3, 6, 7, 8])

        if (di == -1) & (dj == 1):
            ids1 = ID_flat1.take([0, 3, 6, 7, 8])
            ids2 = ID_flat2.take([0, 1, 2, 5, 8])

        for id in ids1:
            dE -= E_mat[II, id]
            dE += E_mat[JJ, id]
        for id in ids2:
            dE -= E_mat[JJ, id]
            dE += E_mat[II, id]
    return 2*dE #count both directions of an interaction
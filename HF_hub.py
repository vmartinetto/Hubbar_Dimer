import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eigh
import scipy as sp

####################### Hubbard Hamiltonian#####################################

def veff(v,U,n):
    '''
    INPUT:
        v: scalar
            The onsite potential
        U: scalar
            The onsite interaction
        n: scalar
            The onsite occupation
    OUTPUT: 
        veff: scalar
            The effective on-site potential
    '''
    return v + U*n

def hub_HF_ham(U,t,v1,v2,n1,n2):
    '''
    INPUT:
        U: scalar
            The charging energy, the on-site interaction term.
        t: scalar
            The hopping term, the kinetic energy.
        v1: scalar
            The onsite potential at site 1.
        v2: scalar
            The onsite potential at site 2.
        n1: scalar
            The onsite occupation of site 1.
        n2: scalar
            The onsite occupation of site 2.
    OUTPUT:
        ham: np.array, size=(2,2)
            The hamiltonian of the HF hubbard dimer.
    '''

    v1eff = veff(v1,U,n1)
    v2eff = veff(v2,U,n2)
    
    ham = np.array([
        [v1eff, -2*t],
        [-2*t, v2eff]
        ])

    return ham

############################Hubbard SCF#############################################

def hub_scf(U,t,dv):

    [n1_op,n2_op] = hub_dens_ops()

    n1 = .5
    n2 = .5
    v1 = dv/2 
    v2 = -dv/2
    diff = 10

    while diff > 1e-4:

        [vals,vecs] = eigh(hub_HF_ham(U,t,v1,v2,n1,n2))

        vec = vecs[:,0]
        n1_temp = np.dot(vec,np.dot(n1_op,vec))
        n2_temp = np.dot(vec,np.dot(n2_op,vec))

        diff = np.abs(n1-n1_temp)+np.abs(n2-n2_temp)

        n1 = n1_temp
        n2 = n2_temp

    return vals[0]

############################solving the ground state hubbard dimer#########################

def hub_2site_dv(U,t,dvs):
    '''
    DEPENDINCIES:
        hub_2site_ham

    CALLER:
        hub_2sit_Udv

    INPUT:
        U: scalar
            The charging energy, the on-site interaction energy.
        t: scalar
            The hopping term, the kinetic energy.
        dvs: np.array, len=ndv
            The delta vs for which to solve the 2-site hubbard model

    OUTPUT:
        gs_eigs: np.array, len=ndv
            A vector of the ground state eigenvalues of the 2-site
            hubbard model for the vector of delta v values.
        gs_vecs: np.array, size=(4,ndv)
            A matrix storing all of the GS eigenvectors for a value of U and t
            as well as a set of dvs as column vectors.
    '''

    # I use d in place of delta becuase I am lazy, no
    # derivatives are done in this function.

    # initalize the vector for gs energy storage 
    # as well as a vector of all dv values
    ndv = len(dvs)
    gs_eigs = np.empty(ndv)
    gs_vecs = np.empty((4,ndv))

    # move through the values of dv and calculate
    # the ground state energy of each 2 site hubbard model
    for i,dv in enumerate(dvs):
        
        [vals, vecs] = eigh(hub_2site_ham(U,t,dv/2,-dv/2))
        gs_eigs[i] = vals[0]
        gs_vecs[:,i] = vecs[:,0]

    return gs_eigs, gs_vecs

def hub_2site_Udv(Us,t,dvs):
    '''
    DEPENDENCIES:
        hub_2site_dv

    INPUT:
        Us: np.array, len=nU
            A vector of the values of the in site interaction for which
            to solve the 2-site hubbard dimer.
        t: scalar
            The fixed hopping term to calculate all the data for.
        dvs: np.array, len=ndv
            A vector of the values of delta v for which to solve the 
            2-site hubbard dimer at each value of U.
    OUTPUT:
        gs_eigs: np.array, size=(ndv,nU)
            A matrix storing the ground state energy of the 2-site
            hubbard dimer. values are stored as column vectors for 
            each value of U.
        gs_vecs: np.array, size=(nU,4,ndv)
            A tensor storing all of the ground state eigenvectors of the hubbard dimer.
            It contains Nu matrices of column vectors for the hubard dimer at a specific
            U and t value while varying over a delta v.
    '''

    # I again use d in place of delta because I am lazy
    # no derivatives are done in this function.

    # Inialize the array to store all of my ground
    # state energies.
    ndv = len(dvs)
    nU = len(Us)
    gs_eigs = np.empty((ndv,nU))
    gs_vecs = np.empty((nU,4,ndv))

    for i,U in enumerate(Us):
        [vals,vecs] = hub_2site_dv(U,t,dvs)
        gs_eigs[:,i] = vals
        gs_vecs[i,:,:] = vecs

    return gs_eigs,gs_vecs

##################Solving all states of the hubbard dimer##############

def hub_2site_dv_all(U,t,dvs):
    '''
    DEPENDINCIES:
        hub_2site_ham

    INPUT:
        U: scalar
            The charging energy, the on-site interaction energy.
        t: scalar
            The hopping term, the kinetic energy.
        dvs: np.array, len=ndv
            The delta vs for which to solve the 2-site hubbard model

    OUTPUT:
        gs_eigs: np.array, len=ndv
            A vector of the ground state eigenvalues of the 2-site
            hubbard model for the vector of delta v values.
    '''

    # I use d in place of delta becuase I am lazy, no
    # derivatives are done in this function.

    # initalize the vector for gs energy storage 
    # as well as a vector of all dv values
    ndv = len(dvs)
    gs_eigs = np.empty((ndv,4))

    # move through the values of dv and calculate
    # the ground state energy of each 2 site hubbard model
    for i,dv in enumerate(dvs):
        
        [vals, vecs] = eigh(hub_2site_ham(U,t,dv/2,-dv/2))
        gs_eigs[i,:] = vals

    return gs_eigs

###############################Hubbard operators#######################

def hub_dens_ops():
    '''
    OUTPUT:
        n1_op: np.array, size=(2,2)
            The operator for calculating the occupation of site 1 in the
            basis of the 2-site hubbard dimer.

        n2_op: np.array, size=(2,2)
            the operator for calculating the occupation of site 2 in the
            basis set ot the 2-site hbbard dimer.
    '''

    n1_op = np.array([
        [1, 0],
        [0, 0]
        ])

    n2_op = np.array([
        [0, 0],
        [0, 1]
        ])

    return n1_op,n2_op

#########################Plotting functions###########################


if __name__ == '__main__':

    dvs = np.linspace(0,20,40)
    U = .2
    t = 1/2
    
    vals = np.empty(len(dvs))

    for i,dv in enumerate(dvs):
        print(i)
        vals[i] = hub_scf(U,t,dv)

    plt.plot(dvs,vals)
    plt.xlim(0,20)
    plt.ylim(-10,5)
    plt.show()

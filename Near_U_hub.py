import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eigh
import scipy as sp
import scipy.optimize as optimize

####################### Hubbard Hamiltonian#####################################

def hub_2site_ham(U,d,t,v1,v2):
    '''
    INPUT:
        U: scalar
            The charging energy, the on-site interaction term.
        t: scalar
            The hopping term, the kinetic energy.
        d: scalar 
            The scaling of nearest neighbor interaction.
        v1: scalar
            The onsite potential at site 1.
        v2: scalar
            The onsite potential at site 2.
        
    OUTPUT:
        ham: np.array, size=(4,4)
            The hamiltonian of the 2-site extended hubbard model at half-filling.
    '''
    
    ham = np.array([
        [2*v1+U, -t, t, 0],
        [-t, U/d, 0, -t],
        [t, 0, U/d, t],
        [0, -t, t, 2*v2+U]
        ])

    return ham

############################solving the ground state hubbard dimer#########################

def hub_2site(U,t,d,dv):
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
        d: scalar 
            The scaling of nearest neighbor interaction.
        dvs: scalar
            The delta v for which to solve the 2-site hubbard model

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

    # move through the values of dv and calculate
    # the ground state energy of each 2 site hubbard model
        
    [vals, vecs] = eigh(hub_2site_ham(U,t,d,dv/2,-dv/2))
    gs_eig = vals[0]
    gs_vec = vecs[:,0]

    return gs_eig, gs_vec

def hub_2site_dv(U,t,d,dvs):
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
        d: scalar 
            The scaling of nearest neighbor interaction.
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
        
        [vals, vecs] = eigh(hub_2site_ham(U,t,d,dv/2,-dv/2))
        gs_eigs[i] = vals[0]
        gs_vecs[:,i] = vecs[:,0]

    return gs_eigs, gs_vecs

def hub_2site_Udv(Us,t,d,dvs):
    '''
    DEPENDENCIES:
        hub_2site_dv

    INPUT:
        Us: np.array, len=nU
            A vector of the values of the in site interaction for which
            to solve the 2-site hubbard dimer.
        t: scalar
            The fixed hopping term to calculate all the data for.
        d: scalar 
            The scaling of nearest neighbor interaction.
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
        [vals,vecs] = hub_2site_dv(U,t,d,dvs)
        gs_eigs[:,i] = vals
        gs_vecs[i,:,:] = vecs

    return gs_eigs,gs_vecs

##################Solving all states of the hubbard dimer##############

def hub_2site_dv_all(U,t,d,dvs):
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
        
        [vals, vecs] = eigh(hub_2site_ham(U,t,d,dv/2,-dv/2))
        gs_eigs[i,:] = vals

    return gs_eigs

###############################Hubbard operators#######################

def hub_dens_ops():
    '''
    OUTPUT:
        n1_op: np.array, size=(4,4)
            The operator for calculating the occupation of site 1 in the
            basis of the 2-site hubbard dimer.

        n2_op: np.array, size=(4,4)
            the operator for calculating the occupation of site 2 in the
            basis set ot the 2-site hbbard dimer.
    '''

    n1_op = np.array([
        [2, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0]
        ])

    n2_op = np.array([
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 2]
        ])

    return n1_op,n2_op

def hub_spin_dens_ops():
    '''
    OUTPUT:
        n1_up: np.array, size=(4,4)
            The operator for calculating the spin up occupation of site 1 in the
            basis of the 2-site hubbard dimer.

        n2_up: np.array, size=(4,4)
            the operator for calculating the spin up occupation of site 2 in the
            basis set ot the 2-site hbbard dimer.
            
        n1_down: np.array, size=(4,4)
            The operator for calculating the spin down occupation of site 1 in the
            basis of the 2-site hubbard dimer.

        n2_down: np.array, size=(4,4)
            the operator for calculating the spin down occupation of site 2 in the
            basis set ot the 2-site hbbard dimer.
    '''

    n1_up = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
        ])

    n2_up = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
        ])
    
    n1_down = np.array([
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0]
        ])

    n2_down = np.array([
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1]
        ])

    return n1_up,n2_up,n1_down,n2_down

def hub_pot_operator(dv):
    '''
    INPUT: 
        dv: scalar
            The difference in the on-site potentials.
    OUTPUT:
        v_op: np.array, size=(4,4)
            The potential engery operator for the 2-site hubbard model
            with half-filling
    '''
    v1 = dv/2
    v2 = -dv/2
    v_op = np.array([
        [2*v1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 2*v2]
        ])

    return v_op

def hub_int_operator(U,d):
    '''
    INPUT:
        U: scalar
            The on-site interaction energy term
    OUTPUT:
        Vee: np.array, size=(4,4)
            The interaction energy operator for the hbbard dimer
    '''
    Vee = np.array([
        [U, 0, 0, 0],
        [0, U/d, 0, 0],
        [0, 0, U/d, 0],
        [0, 0, 0, U]
        ])
    return Vee

def hub_kin_operator(t):
    '''
    INPUT:
        t: scalar
            The hopping term
    OUTPUT:
        T: np.array, size=(4,4)
            The kinetic energy operator for the hubbard dimer at half 
            filling
    '''
    T = np.array([
        [0, -t, t, 0],
        [-t, 0, 0, -t],
        [t, 0, 0, t],
        [0, -t, t, 0]
        ])
    return T

def hub_non_int_kin_operator(t):
    '''
    INPUT:
        t: scalar
            The hopping term
    OUTPUT:
        T: np.array, size=(4,4)
            The kinetic energy operator for the non-interacting
            hubbard dimer at half filling
    '''
    n1n2 = np.array([
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0]
        ])
    def operator(vec):
        val = np.dot(vec,np.dot(n1n2,vec))
        val = -2*t*np.sqrt(val)
        return val
    return operator

def hub_EHX_operator(U):
    '''
    INPUT:
        U: scalar
            The on-site interaction energy term
    OUTPUT:
        EHX: np.array, size=(4,4)
            The Hartree Exchange energy operator for the hbbard dimer
    '''
    EHX = np.array([
        [U, 0, 0, 0],
        [0, U/2, 0, 0],
        [0, 0, U/2, 0],
        [0, 0, 0, U]
        ])
    return EHX

#########################KS Functions###################################

def Ts(n1,t=.5):
    '''
    INPUT:
        n1: scalar
            The occupation on site 1
        t: scalar, default = 1/2
            The hopping term for the described system
    OUTPUT:
        Ts: scalar
            The non-interacting kintic energy for a given occupation
            of site 1.
    '''

    n2 = 2-n1
    Ts = -2*t*np.sqrt(n1*n2)

    return Ts

def E_HX(n1,U):
    n2 = 2-n1
    return (U/4)*(n1**2+n2**2)

def E_H(n1,U):
    n2 = 2-n1
    return (U/2)*(n1**2+n2**2)

def E_HX_dn(dn,U):
    return (U/2)*(1+(dn/2)**2)

####################Adiabatic connection functions####################

def dn_search_func_gen(U,t,d,target_dn):
    '''
    Takes a U and t value for the zero-temp Hubbard dimer and creates an function
    that is the shifted occupation sigmoid with a target delta n as the root.
    
    INPUT
        U: scalar
            The on-site interaction term
        t: scalar
            The hopping parameter
        d: scalar 
            The scaling of nearest neighbor interaction.
        target_dn: scalar
            The target delta n to achieve
    OUTPUT
        func: python function
            A function that takes a site potnetial difference and returns the shifted 
            site occupation differnece.
    '''

    def func(dv):
        
        n1,n2 = hub_dens_ops()
        dn = n2-n1
        
        eig, vec = hub_2site(U,t,d,dv)
        calc_dn = np.dot(vec.T,np.dot(dn,vec))
        
        return calc_dn-target_dn
    
    return func

def hub_Uc(t,lam,U,target_dn,EHX):
    '''
    Takes multiple values and comutes Uc at a target delta n for those values.
    
    INPUT
        t: scalar
            The hopping parameter.
        lam: scalar
            the interaction strength scaling paramter.
        U: scalar
            The interaction strength of the full system.
        target_dn: scalar
            the on-site occupation differnece that is to remain fixed.
        E_HX: scalar
            The Hartree Exchange energy of the non-interacting system at the 
            targert delta n.
    
    OUTPUT
        Uc: scalar
            The potnetial correlation energy at given values above.
            
    '''
    func = dn_search_func_gen(lam*U,t,target_dn)
    dv = optimize.bisect(func,-10000000000000,100000000000000,xtol=2e-12)
        
    Vee_op = hub_int_operator(lam*U)
        
    eig, vec = hub_2site(lam*U,t,dv)
        
    return ((np.dot(vec.T,np.dot(Vee_op,vec))-lam*EHX)/lam)

def hub_Uxc(t,lam,U,target_dn,EH):
    '''
    Takes multiple values and comutes Uc at a target delta n for those values.
    
    INPUT
        t: scalar
            The hopping parameter.
        lam: scalar
            the interaction strength scaling paramter.
        U: scalar
            The interaction strength of the full system.
        target_dn: scalar
            the on-site occupation differnece that is to remain fixed.
        E_H: scalar
            The Hartree energy of the non-interacting system at the 
            targert delta n.
    
    OUTPUT
        Uxc: scalar
            The potnetial exchange correlation energy at given values above.
            
    '''
    func = dn_search_func_gen(lam*U,t,target_dn)
    dv = optimize.bisect(func,-1000000,1000000,xtol=2e-12)
        
    Vee_op = hub_int_operator(lam*U)
        
    eig, vec = hub_2site(lam*U,t,dv)
        
    return ((np.dot(vec.T,np.dot(Vee_op,vec))-lam*EH)/lam)

def hub_Uc_lams(t,lams,U,target_dn,EHX):
    Ucs = np.empty(len(lams))
    for i,lam in enumerate(lams):
        Ucs[i] = hub_Uc(t,lam,U,target_dn,EHX)
    return Ucs

def hub_Uxc_lams(t,lams,U,target_dn,EH):
    Uxcs = np.empty(len(lams))
    for i,lam in enumerate(lams):
        Uxcs[i] = hub_Uxc(t,lam,U,target_dn,EH)
    return Uxcs

def hub_Uc_lams_Us_dvs_overU(t,lams,Us,dvs):
    
    n1,n2 = hub_dens_ops()
    
    ab_eigs, ab_vecs = hub_2site_Udv(Us,t,dvs)

    Ucs = np.empty((len(dvs),len(lams),len(Us)))

    for i,U in enumerate(Us):
        for j,dv in enumerate(dvs):
    
            occ1 = np.dot(ab_vecs[i,:,j].T,np.dot(n1,ab_vecs[i,:,j]))
            target_dn = (2-occ1)-occ1
    
            EHX = E_HX(occ1,U)
    
            Ucs[j,:,i] = hub_Uc_lams(t,lams,U,target_dn,EHX)/U
        
    return Ucs

def hub_Uxc_lams_Us_dvs_overU(t,lams,Us,dvs):
    
    n1,n2 = hub_dens_ops()
    
    ab_eigs, ab_vecs = hub_2site_Udv(Us,t,dvs)

    Uxcs = np.empty((len(dvs),len(lams),len(Us)))

    for i,U in enumerate(Us):
        for j,dv in enumerate(dvs):
    
            occ1 = np.dot(ab_vecs[i,:,j].T,np.dot(n1,ab_vecs[i,:,j]))
            target_dn = (2-occ1)-occ1
    
            EH = E_H(occ1,U)
    
            Uxcs[j,:,i] = hub_Uxc_lams(t,lams,U,target_dn,EH)/U
        
    return Uxcs

def hub_Uc_lams_Us_dvs(t,lams,Us,dvs):
    
    n1,n2 = hub_dens_ops()
    
    ab_eigs, ab_vecs = hub_2site_Udv(Us,t,dvs)

    Ucs = np.empty((len(dvs),len(lams),len(Us)))

    for i,U in enumerate(Us):
        for j,dv in enumerate(dvs):
    
            occ1 = np.dot(ab_vecs[i,:,j].T,np.dot(n1,ab_vecs[i,:,j]))
            target_dn = (2-occ1)-occ1
    
            EHX = E_HX(occ1,U)
    
            Ucs[j,:,i] = hub_Uc_lams(t,lams,U,target_dn,EHX)
        
    return Ucs

def hub_Uxc_lams_Us_dvs(t,lams,Us,dvs):
    
    n1,n2 = hub_dens_ops()
    
    ab_eigs, ab_vecs = hub_2site_Udv(Us,t,dvs)

    Uxcs = np.empty((len(dvs),len(lams),len(Us)))

    for i,U in enumerate(Us):
        for j,dv in enumerate(dvs):
    
            occ1 = np.dot(ab_vecs[i,:,j].T,np.dot(n1,ab_vecs[i,:,j]))
            target_dn = (2-occ1)-occ1
    
            EH = E_H(occ1,U)
    
            Uxcs[j,:,i] = hub_Uxc_lams(t,lams,U,target_dn,EH)
        
    return Uxcs
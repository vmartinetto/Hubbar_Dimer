import numpy as np
from numpy.linalg import eigh
import scipy as sp
import scipy.optimize as optimize
from decimal import Decimal

####################### Hubbard Hamiltonians#####################################

def hub_2site_1part(U,t,v1,v2):
    '''
    The hamiltonian of the hubbard model in the basis [|1u>,|1d>,|2u>,|2d>] 
    INPUT:
        U: scalar
            The charging energy, the on-site interaction term.
        t: scalar
            The hopping term, the kinetic energy.
        v1: scalar
            The onsite potential at site 1.
        v2: scalar
            The onsite potential at site 2.
    OUTPUT:
        ham: np.array, size=(4,4)
            The hamiltonian of the 2-site hubbard model at with one particle.
    '''
    
    #print(v1)
    
    ham = np.array([
        [v1, 0, -t, 0],
        [0, v1, 0, -t],
        [-t, 0, v2, 0],
        [0, -t, 0, v2]
        ])

    return ham

def hub_2site_2part(U,t,v1,v2):
    '''
    The basis for this hamiltonian is [|1u1d>,|1u2d>,|1d2u>,|2u2d>,|1u2u>,|1d2d>]
    
    INPUT:
        U: scalar
            The charging energy, the on-site interaction term.
        t: scalar
            The hopping term, the kinetic energy.
        v1: scalar
            The onsite potential at site 1.
        v2: scalar
            The onsite potential at site 2.
    OUTPUT:
        ham: np.array, size=(4,4)
            The hamiltonian of the 2-site hubbard model at half-filling with both anit-parallel and parallel spin.
    '''
    
    ham = np.array([
        [2*v1+U, -t, t, 0, 0, 0],
        [-t, 0, 0, -t, 0, 0],
        [t, 0, 0, t, 0, 0],
        [0, -t, t, 2*v2+U, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
        ])

    return ham

def hub_2site_3part(U,t,v1,v2):
    '''
    The basis for this hamiltonian is [|1u1d2d>,|1u1d2u>,|1u2u2d>,|1d2u2d>]
    
    INPUT:
        U: scalar
            The charging energy, the on-site interaction term.
        t: scalar
            The hopping term, the kinetic energy.
        v1: scalar
            The onsite potential at site 1.
        v2: scalar
            The onsite potential at site 2.
    OUTPUT:
        ham: np.array, size=(4,4)
            The hamiltonian of the 2-site hubbard model at with 3 particles.
    '''
    
    ham = np.array([
        [2*v1+v2+U, 0, 0, -t],
        [0, U+2*v1+v2, -t, 0],
        [0, -t, U+2*v2+v1, 0],
        [-t, 0, 0, 2*v2+v1+U]
        ])

    return ham

def hub_2site_4part(U,t,v1,v2):
    '''
    The basis for this hamiltonian is |1u1d2u2d>
    
    INPUT:
        U: scalar
            The charging energy, the on-site interaction term.
        t: scalar
            The hopping term, the kinetic energy.
        v1: scalar
            The onsite potential at site 1.
        v2: scalar
            The onsite potential at site 2.
    OUTPUT:
        ham: np.array, size=(4,4)
            The hamiltonian of the 2-site hubbard model at with 4 particles.
    '''
    
    ham = 2*U

    return ham

######################## occupation operators ######################################

def hub1_dens_ops():
    '''
    The occupation operators in the basis [|1u>,|1d>,|2u>,|2d>]
    
    OUTPUT:
        n1_op: np.array, size=(4,4)
            The operator for calculating the occupation of site 1 in the
            basis of the 2-site hubbard model with one particle.

        n2_op: np.array, size=(4,4)
            the operator for calculating the occupation of site 2 in the
            basis set ot the 2-site hubbard dimer with one particle.
        
        n1_op-n2_op: np.array, size=(4,4)
            the operator for calculating the difference in site occuation of 
            the 2-site hubbard model with one-particle
    '''

    n1_op = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
        ])

    n2_op = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
        ])

    return n1_op,n2_op,n2_op-n1_op

def hub2_dens_ops():
    '''
    The occupation operators in the basis [|1u1d>,|1u2d>,|1d2u>,|2u2d>,|1u2u>,|1d2d>]
    
    OUTPUT:
        n1_op: np.array, size=(6,6)
            The operator for calculating the occupation of site 1 in the
            basis of the 2-site hubbard dimer.

        n2_op: np.array, size=(6,6)
            the operator for calculating the occupation of site 2 in the
            basis set ot the 2-site hbbard dimer.
            
        n1_op-n2_op: np.array, size=(6,6)
            the operator for calculating the difference in site occuation of 
            the 2-site hubbard model with two-particles
    '''

    n1_op = np.array([
        [2, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
        ])

    n2_op = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
        ])

    return n1_op,n2_op,n2_op-n1_op

def hub3_dens_ops():
    '''
    The occupation operators in the basis [|1u1d2d>,|1u1d2u>,|1u2u2d>,|1d2u2d>]
    
    OUTPUT:
        n1_op: np.array, size=(4,4)
            The operator for calculating the occupation of site 1 in the
            basis of the 2-site hubbard dimer.

        n2_op: np.array, size=(4,4)
            the operator for calculating the occupation of site 2 in the
            basis set ot the 2-site hbbard dimer.
            
        n1_op-n2_op: np.array, size=(4,4)
            the operator for calculating the difference in site occuation of 
            the 2-site hubbard model with three-particles
    '''

    n1_op = np.array([
        [2, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
        ])

    n2_op = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 2]
        ])

    return n1_op,n2_op,n2_op-n1_op

######################## Interaction Energy operators ##################################

def hub1_U_op(U):
    '''
    The interaction energy operator in the basis [|1u>,|1d>,|2u>,|2d>]
    
    OUTPUT:
        U_op: np.array, size=(4,4)
            The operator for calculating the interaction energy of each state in the
            basis of the 2-site hubbard model with one particle.

    '''

    U_op = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
        ])

    return U_op

def hub2_U_op(U):
    '''
    The interaction energy operator in the basis [|1u1d>,|1u2d>,|1d2u>,|2u2d>,|1u2u>,|1d2d>]
    
    OUTPUT:
        U_op: np.array, size=(6,6)
            The interaction energgy operator in the basis of the 2-site hubbard dimer.

    '''

    U_op = np.array([
        [U, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, U, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
        ])

    return U_op

def hub3_U_op(U):
    '''
    The interaction operator in the basis [|1u1d2d>,|1u1d2u>,|1u2u2d>,|1d2u2d>]
    
    OUTPUT:
        U_op: np.array, size=(4,4)
            The operator for calculating the interaction energy of site 1 in the
            basis of the 2-site hubbard dimer.

    '''

    U_op = np.array([
        [U, 0, 0, 0],
        [0, U, 0, 0],
        [0, 0, U, 0],
        [0, 0, 0, U]
        ])

    return U_op

######################## Kinetic energy operators ##############################

def hub1_kin_op(t):
    '''
    The kinetic energy operator in the basis [|1u>,|1d>,|2u>,|2d>]
    
    OUTPUT:
        kin_op: np.array, size=(4,4)
            The operator for calculating the kinetic energy of each state in the
            basis of the 2-site hubbard model with one particle.

    '''

    kin_op = np.array([
        [0, 0, -t, 0],
        [0, 0, 0, -t],
        [-t, 0, 0, 0],
        [0, -t, 0, 0]
        ])

    return kin_op

def hub2_kin_op(t):
    '''
    The kinetic energy operator in the basis [|1u1d>,|1u2d>,|1d2u>,|2u2d>,|1u2u>,|1d2d>]
    
    OUTPUT:
        kin_op: np.array, size=(6,6)
            The kinetic energgy operator in the basis of the 2-site hubbard dimer.

    '''

    kin_op = np.array([
        [0, -t, t, 0, 0, 0],
        [-t, 0, 0, -t, 0, 0],
        [t, 0, 0, t, 0, 0],
        [0, -t, t, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
        ])

    return kin_op

def hub3_kin_op(t):
    '''
    The kinetic operator in the basis [|1u1d2d>,|1u1d2u>,|1u2u2d>,|1d2u2d>]
    
    OUTPUT:
        kin_op: np.array, size=(4,4)
            The operator for calculating the kinetic energy of site 1 in the
            basis of the 2-site hubbard dimer.

    '''

    kin_op = np.array([
        [0, 0, 0, -t],
        [0, 0, -t, 0],
        [0, -t, 0, 0],
        [-t, 0, 0, 0]
        ])

    return kin_op

######################## External potential operator ###############################

def hub1_V_op(v1,v2):
    '''
    The external potential energy operator in the basis [|1u>,|1d>,|2u>,|2d>]
    
    OUTPUT:
        V_op: np.array, size=(4,4)
            The operator for calculating the external potential energy of each state in the
            basis of the 2-site hubbard model with one particle.

    '''

    V_op = np.array([
        [v1, 0, 0, 0],
        [0, v1, 0, 0],
        [0, 0, v2, 0],
        [0, 0, 0, v2]
        ])

    return V_op

def hub2_V_op(v1,v2):
    '''
    The external potential operator in the basis [|1u1d>,|1u2d>,|1d2u>,|2u2d>,|1u2u>,|1d2d>]
    
    OUTPUT:
        kin_op: np.array, size=(6,6)
            The external potential energgy operator in the basis of the 2-site hubbard dimer.

    '''

    V_op = np.array([
        [2*v1, 0, 0, 0, 0, 0],
        [0, v1+v2, 0, 0, 0, 0],
        [0, 0, v1+v2, 0, 0, 0],
        [0, 0, 0, 2*v2, 0, 0],
        [0, 0, 0, 0, v1+v2, 0],
        [0, 0, 0, 0, 0, v1+v2]
        ])

    return V_op

def hub3_V_op(v1,v2):
    '''
    The external potential operator in the basis [|1u1d2d>,|1u1d2u>,|1u2u2d>,|1d2u2d>]
    
    OUTPUT:
        kin_op: np.array, size=(4,4)
            The operator for calculating the external potential energy of site 1 in the
            basis of the 2-site hubbard dimer.

    '''

    V_op = np.array([
        [2*v1+v2, 0, 0, 0],
        [0, 2*v1+v2, 0, 0],
        [0, 0, v1+2*v2, 0],
        [0, 0, 0, v1+2*v2]
        ])

    return V_op

######################## Interacting Partition Func ################################

def Partition(U,t,v1,v2,mu,taus):
    '''
    A function for calculating the partiontion function and grand cannonical potential
    of the hubbard model given specified parameters.
    
    INPUT:
        U: scalar
            The charging energy, the on-site interaction term.
        t: scalar
            The hopping term, the kinetic energy.
        v1: scalar
            The onsite potential at site 1.
        v2: scalar
            The onsite potential at site 2.
        mu: scalar
            The chemical potential of the system.
        taus: np.array
            A list of tau values to calculate the partition function of the system at.
    OUTPUT:
        Zgcs: np.array
            A list of the partion functions for the hubbard model with parameters specified.
        Oms: np.array
            A list of the grand cannonical potentail values at each temp tau
    '''
    
    # solve for the eigenvectors and values at the specified U, t ,v1, v2 vals
    onevals,onevecs = eigh(hub_2site_1part(U,t,v1,v2))
    twovals,twovecs = eigh(hub_2site_2part(U,t,v1,v2))
    threevals,threevecs = eigh(hub_2site_3part(U,t,v1,v2))
    fourvals = 2*U
    
    # calculate the part not dependent on tau so tau can be looped over and
    # eigenvalues don't have to be re calculated
    ZG_list = np.array([0])
    ZG_list = np.append(ZG_list,mu*1 - onevals)
    ZG_list = np.append(ZG_list,mu*2 - twovals)
    ZG_list = np.append(ZG_list,mu*3 - threevals)
    ZG_list = np.append(ZG_list,mu*4 - fourvals)
    ZG_list = np.array([Decimal(val) for val in ZG_list])
    
    # intialize the partion function storage and calculate
    # calculate the grand cannonical potential as well
    Zgcs = np.empty(len(taus))
    Oms = np.empty(len(taus))
    for i,tau in enumerate(taus):
        tau = Decimal(tau)
        Zg = np.sum(np.exp(ZG_list/tau))
        Zgcs[i] = Zg
        Oms[i] = -tau*Zg.ln()
    return Zgcs, Oms

def Expectation(U,t,v1,v2,mu,taus,operator='delta_n'):
    '''
    A function for calculating the expectation value of the thermal hubbard model at
    temp tau for a set of given parameters.
    
    INPUT:
        U: scalar
            The charging energy, the on-site interaction term.
        t: scalar
            The hopping term, the kinetic energy.
        v1: scalar
            The onsite potential at site 1.
        v2: scalar
            The onsite potential at site 2.
        mu: scalar
            The chemical potential of the system.
        taus: np.array
            A list of tau values to calculate the partition function of the system at.
    OUTPUT:
        delta_ns:
    '''
    
    # Initialize which operator to calculate the expectation value of
    if operator == 'delta_n':
        _,_,op1 = hub1_dens_ops()
        _,_,op2 = hub2_dens_ops()
        _,_,op3 = hub3_dens_ops()
        op4 = 0
    elif operator == 'kinetic':
        op1 = hub1_kin_op(t)
        op2 = hub2_kin_op(t)
        op3 = hub3_kin_op(t)
        op4 = 0
    elif operator == 'interaction':
        op1 = hub1_U_op(U)
        op2 = hub2_U_op(U)
        op3 = hub3_U_op(U)
        op4 = 2*U
    elif operator == 'ext_potential':
        op1 = hub1_V_op(v1,v2)
        op2 = hub2_V_op(v1,v2)
        op3 = hub3_V_op(v1,v2)
        op4 = 2*v1+2*v2
        
    # solve for the eigenvectors and values at the specified U, t ,v1, v2 vals
    onevals,onevecs = eigh(hub_2site_1part(U,t,v1,v2))
    twovals,twovecs = eigh(hub_2site_2part(U,t,v1,v2))
    threevals,threevecs = eigh(hub_2site_3part(U,t,v1,v2))
    fourvecs,fourvals = np.array([1]),2*U
    
    # calculate the part not dependent on tau so tau can be looped over and
    # eigenvalues don't have to be re calculated
    ZG_list = np.array([0])
    ZG_list = np.append(ZG_list,mu*1 - onevals)
    ZG_list = np.append(ZG_list,mu*2 - twovals)
    ZG_list = np.append(ZG_list,mu*3 - threevals)
    ZG_list = np.append(ZG_list,mu*4 - fourvals)
    ZG_list = np.array([Decimal(val) for val in ZG_list])
    
    # solve for the list of all expectation values in a manner similar to ZG_list
    exp_list = np.array([0])
    exp_list = np.append(exp_list,np.diagonal(np.dot(onevecs.T,np.dot(op1,onevecs))))
    exp_list = np.append(exp_list,np.diagonal(np.dot(twovecs.T,np.dot(op2,twovecs))))
    exp_list = np.append(exp_list,np.diagonal(np.dot(threevecs.T,np.dot(op3,threevecs))))
    exp_list = np.append(exp_list,op4)
    exp_list = np.array([Decimal(val) for val in exp_list])
    
    # intialize the expectation storage and calculate
    # the averaged expectation value
    expts = np.empty(len(taus))
    for i,tau in enumerate(taus):
        tau = Decimal(tau)
        Zg = np.sum(np.exp(ZG_list/tau))
        expts[i] = np.sum(exp_list*np.exp(ZG_list/tau))/Zg
    return expts
    
######################################### 2 particle functions #################################

def Thermal_2particle_Hubbard(U,t,dv,taus):
    '''
    A function for calculating the partiontion function, grand cannonical potential,
    and the free energy of the hubbard dimer given specified parameters.
    
    INPUT:
        U: scalar
            The charging energy, the on-site interaction term.
        t: scalar
            The hopping term, the kinetic energy.
        dv: scalar
            The difference in the on-site potnetials.
        taus: np.array
            A list of tau values to calculate the partition function of the system at.
    OUTPUT:
        Zgcs: np.array
            A numpy array of the partion functions for the hubbard model with parameters specified.
        Oms: np.array
            A numpy array of the grand cannonical potentail values at each temp tau
        As: np.array
            A numpy array of the free energies of the Hubbard dimer.
    '''
    mu = U/2
    Zgcs,Oms = Partition(U,t,dv/2,-dv/2,mu,taus)
    As = mu*2+Oms
    return Zgcs,Oms,As

def Thermal_2particle_Hubbard_dvs(U,t,dvs,taus):
    '''
    A function looping over a set of dvs and calling the base function 
    Thermal_2particle_Hubbard()
    
    INPUT:
        U: scalar
            The charging energy, the on-site interaction term.
        t: scalar
            The hopping term, the kinetic energy.
        dvs: np.array
            A numpy vector listing a set of on-site potnetial differences.
        taus: np.array
            A numpy vector of tau values to calculate the partition function of the system at.
    OUTPUT:
        Zgcs: np.array
            A numpy array of the partion functions for the hubbard model with parameters specified.
        Oms: np.array
            A numpy array of the grand cannonical potentail values at each temp tau
        As: np.array
            A numpy array of the free energies of the Hubbard dimer.
    '''
    Zgcs = np.empty((len(taus),len(dvs)))
    Oms = np.empty((len(taus),len(dvs)))
    As = np.empty((len(taus),len(dvs)))
    for i,dv in enumerate(dvs):
        Zgcs[:,i],Oms[:,i],As[:,i] = Thermal_2particle_Hubbard(U,t,dv,taus)
    return Zgcs,Oms,As

def Thermal_2particle_Hubbard_deltan(U,t,dv,taus):
    '''
    A function for calculating the difference in site occupation for the 
    2 particle 2 site thermal hubbard model.
    
    INPUT:
        U: scalar
            The charging energy, the on-site interaction term.
        t: scalar
            The hopping term, the kinetic energy.
        dv: scalar
            The difference in the on-site potnetials.
        taus: np.array
            A list of tau values to calculate the partition function of the system at.
    OUTPUT:
        expts: np.array
            The delta n values for a given dv,U, and T across a range of tau
    '''
    mu = U/2
    expts = Expectation(U,t,dv/2,-dv/2,mu,taus,operator='delta_n')
    return expts

def Thermal_2particle_Hubbard_deltan_dvs(U,t,dvs,taus):
    '''
    A function for calculating the difference in site occupation for the 
    2 particle 2 site thermal hubbard model. wraps around another function.
    
    INPUT:
        U: scalar
            The charging energy, the on-site interaction term.
        t: scalar
            The hopping term, the kinetic energy.
        dv: scalar
            The difference in the on-site potnetials.
        taus: np.array
            A list of tau values to calculate the partition function of the system at.
    OUTPUT:
        expts: np.array, shape(ntau,ndv)
            The delta n values for a given U and t across a range of tau and dvs
    '''
    expts = np.empty((len(taus),len(dvs)))
    for i,dv in enumerate(dvs):
        expts[:,i] = Thermal_2particle_Hubbard_deltan(U,t,dv,taus)
    return expts

def Thermal_2particle_Hubbard_kinetic(U,t,dv,taus):
    '''
    A function for calculating the kinetic energy for the 
    2 particle 2 site thermal hubbard model.
    
    INPUT:
        U: scalar
            The charging energy, the on-site interaction term.
        t: scalar
            The hopping term, the kinetic energy.
        dv: scalar
            The difference in the on-site potnetials.
        taus: np.array
            A list of tau values to calculate the partition function of the system at.
    OUTPUT:
        expts: np.array
            The kinetic values for a given dv,U, and T across a range of tau
    '''
    mu = U/2
    expts = Expectation(U,t,dv/2,-dv/2,mu,taus,operator='kinetic')
    return expts

def Thermal_2particle_Hubbard_kinetic_dvs(U,t,dvs,taus):
    '''
    A function for calculating the kinetic energy for the 
    2 particle 2 site thermal hubbard model.
    
    INPUT:
        U: scalar
            The charging energy, the on-site interaction term.
        t: scalar
            The hopping term, the kinetic energy.
        dv: scalar
            The difference in the on-site potnetials.
        taus: np.array
            A list of tau values to calculate the partition function of the system at.
    OUTPUT:
        expts: np.array
            The kinetic values for a given dv,U, and T across a range of tau
    '''
    expts = np.empty((len(taus),len(dvs)))
    for i,dv in enumerate(dvs):
        expts[:,i] = Thermal_2particle_Hubbard_kinetic(U,t,dv,taus)
    return expts

def Thermal_2particle_Hubbard_interaction(U,t,dv,taus):
    '''
    A function for calculating the kinetic energy for the 
    2 particle 2 site thermal hubbard model.
    
    INPUT:
        U: scalar
            The charging energy, the on-site interaction term.
        t: scalar
            The hopping term, the kinetic energy.
        dv: scalar
            The difference in the on-site potnetials.
        taus: np.array
            A list of tau values to calculate the partition function of the system at.
    OUTPUT:
        expts: np.array
            The kinetic values for a given dv,U, and T across a range of tau
    '''
    mu = U/2
    expts = Expectation(U,t,dv/2,-dv/2,mu,taus,operator='interaction')
    return expts

def Thermal_2particle_Hubbard_interaction_dvs(U,t,dvs,taus):
    '''
    A function for calculating the kinetic energy for the 
    2 particle 2 site thermal hubbard model.
    
    INPUT:
        U: scalar
            The charging energy, the on-site interaction term.
        t: scalar
            The hopping term, the kinetic energy.
        dv: scalar
            The difference in the on-site potnetials.
        taus: np.array
            A list of tau values to calculate the partition function of the system at.
    OUTPUT:
        expts: np.array
            The kinetic values for a given dv,U, and T across a range of tau
    '''
    expts = np.empty((len(taus),len(dvs)))
    for i,dv in enumerate(dvs):
        expts[:,i] = Thermal_2particle_Hubbard_interaction(U,t,dv,taus)
    return expts

############################### Adiabatic connection functions ###################

def Thermal_dn_search_func_gen(U,t,tau,target_dn):
    '''
    A function for calculating the expectation value of the thermal hubbard model at
    temp tau for a set of given parameters.
    
    INPUT:
        U: scalar
            The charging energy, the on-site interaction term.
        t: scalar
            The hopping term, the kinetic energy.
        mu: scalar
            The chemical potential of the system.
        taus: np.array
            A list of tau values to calculate the partition function of the system at.
        target_dn: scalar
            the target on-site density diffenrce.
    OUTPUT:
        func: python function
            function takes dv and returns the site occupation difference
    '''
    
    mu = U/2
    taus = [tau]
    
    def func(dv):
        '''
        A function that is initialized through dn_search_func_gen that takes in the on-site
        potential diffence and outputs the on-site occupation diffence. Will be used to find dv which returns
        the correct target dn.
        
        INPUT:
            dv: scalar
                on-site potentail difference to test
        OUTPUT
            expt-target_dn: scalar
                should return zero when dv gives the target on-site density diffence.
        '''
        #print('func dv:',dv)
        expt = Expectation(U,t,dv/2,-dv/2,mu,taus,operator='delta_n')
        #print('expectation:',expt)
        return float(expt-target_dn)
    
    return func

def Thermal_hub_Uc(t,lam,U,tau,target_dn,EHX,dv_guess):
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
    func = Thermal_dn_search_func_gen(lam*U,t,tau,target_dn)
    #dv_guess = 10*lam*U*target_dn
    #if lam < 1:
    #    dv = optimize.bisect(func,-100,100,xtol=2e-12)
    #else:
    #    dv = optimize.bisect(func,-100*np.sqrt(lam/10),100*np.sqrt(lam/10),xtol=2e-12)
    dv = optimize.newton(func,dv_guess,maxiter=100)
        
    Vee = Thermal_2particle_Hubbard_interaction(lam*U,t,dv,[tau])
        
    return ((Vee-lam*EHX)/lam),dv
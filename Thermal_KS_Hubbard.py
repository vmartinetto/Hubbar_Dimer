# numbered equations come from the paper Phys rev B. 93 245131 (2016)

import numpy as np
import scipy.optimize as optimize

############################ delta_n equations ############################

def xf(dv,t):
    '''
    Calculates the variable x based on the inline equation in the paper
    
    INPUT
        dv: scalar or np.array
            The on-site single particle potential differnece for the hubbard dimer.
        t: scalar or np.array
            The hopping term that is shared between the interacting and non-interacting 
            system.
    OUTPUT:
        x: scalar or np.array
            The on-site single particle potential differnece scaled by the hopping term.
    '''
    return dv/(2*t)

def thetaf(x):
    '''
    Calculates the phi term in the in-line equations in the paper
    
    INPUT
        x: scalar or np.array
            The single particle on-site potnetial differnece scaled by the hopping term.
    OUTPUT
        theta: scalar or np.array
            Another constant from the paper that measures the distance of the scaled
            dv term from 0.
    '''
    return np.arcsin(x/np.sqrt(1+x**2))

def alphaf(tau,theta):
    '''
    Calculates the alpha term from the in-line equations in the paper.
    
    INPUT
        tau: scalar or np.array
            The temprature of the system in the energy units of the system
        theta: scalar or np.array
            The measure of dv's distacne from zero.
    OUTPUT
        alpha: scalar or np.array
            Another distance measure which includes the temprature of the system.
    '''
    return 1/(4*tau*np.cos(theta))

def del_n_calc(dv,t,tau):
    '''
    Calculates what delta_n should be for a given single-particle on-site potnential
    diffrence, hopping paramter t, and temprature tau. No U because the non-interacting system.
    
    INPUT
        dv: scalar or np.array
            The on-site single particle potential differnece for the hubbard dimer.
        t: scalar or np.array
            The hopping term that is shared between the interacting and non-interacting 
            system.
        tau: scalar or np.array
            The temprature of the system in the energy units of the system
    OUTPUT
        del_n: scalar or np.array
            The on-site occupation difference of the interacting system from a 
            non-interacting hubbard dimer.
    '''
    x = xf(dv,t)
    theta = thetaf(x)
    alpha = alphaf(tau,theta)
    return -2*np.sin(theta)*np.tanh(alpha)

def del_n_root(t,tau,target_dn):
    '''
    Generates a function that caluclates the on-site occupation differnce and subtracts off
    the target delta n. Thi function is for use with root finding methods.
    
    INPUT:
        t: scalar
            The target hopping term to find the correct single-particle on-site potential 
            differnce for.
        tau: scalar
            The target temprature to find the correct single-particle on-site potential 
            differnce for.
        target_dn: scalar
            The target dn of the interacting hubbard dimer to find with the non-interacting 
            hubbard dimer dv.
    OUTPUT:
        del_n: python function
            A function that calctates the differnce between dn for a specific single-paritcle 
            on-site potential difference and a target dn.
    '''
    def del_n(dv):
        '''
        A python function generated to calculate delta n at a specific t and tau value
        given a variable dv. A target delta n is then subtracted to make the target the 
        root of the function. Returned by the del_n_root function and used to find the correct
        non-interacting dv that makes delta n in the interacting case.
        
        INPUT
            dv: scalar or np.array
            The on-site single particle potential differnece for the hubbard dimer.
            
        OUTPUT
            del_n-target_dn: scalar
                The distance from the target del_n. if the value returned is zero then you have the 
                correct single particle dv that recreates your target delta n of the interacting system
                at temp tau with hooping parameter t.
        '''
        del_n = del_n_calc(dv,t,tau)
        return del_n-target_dn
    return del_n

########################### Root finding with bisection ##################

def find_dvks(t,taus,delta_ns):
    '''
    takes a fixed hopping parameter t, a set of temps tau, a set of delta_ns at each temp
    tau in taus and calculates a non-interacting potential that recreates that delta_n at the 
    same temprature tau.
    
    INPUT
        t: scalar
            The hopping paramter of the system.
        taus: np.array, len = N
            A set of tempratures tau in an np.array of length N
        delta_ns: np.array, len=N
            A set of delta_ns at each temp tau also of length N
            
    OUTPUT
        dvks np.array, len = N
            A set of non-interacting on-site potnetial differences that recreate the target densities
            at each given tmeprature tau.
    '''
    dvks = np.empty(len(delta_ns))
    for i,tau in enumerate(taus):
        func = del_n_root(t,tau,delta_ns[i])
        dvks[i] = optimize.bisect(func,-40,40)
    return dvks

############################ KS Energies ##################################

def entropy(dvks,t,tau):
    '''
    Equation 21 of the paper quoted. Takes a on-site potential dvks, hopping parameter t,
    and temprature tau and returns the entropy of the non-interacting entropy of that system.
    
    INPUT:
        dvks: scalar or np.array
            The on-site single particle potential differnece for the hubbard dimer.
        t: scalar or np.array
            The hopping term that is shared between the interacting and non-interacting 
            system.
        tau: scalar or np.array
            The temprature of the system in the energy units of the system
    
    OUTPUT:
        S: scalar or np.array
            The entropy of the non-interacting system at given system parameters.
    '''
    
    x = xf(dvks,t)
    theta = np.arcsin(x/np.sqrt(1+x**2))
    alpha = 1/(4*tau*np.cos(theta))
    
    return 4*np.log(2*np.cosh(alpha))-4*alpha*np.tanh(alpha)

def UHX(U,delta_n):
    '''
    Calculates UHX based on equation 18 and the information that Ex is -.5*UH
    
    INPUT:
        U: Scalar
            The interaction strength of the interacting system.
        delta_n: scalar or np.array
            The density differnce of the interacting system.

    OUTPUT:
        UHX: scalar or np.array
            The hartree exchange energy of the non-interacting system.
    '''
    return .5*U*(1+(delta_n**2/4))

def Ts(t,tau,dvks):
    '''
    Calculates the KS kinetic energy based on the exact equations derived in
    the mathematica notebooks in this folder.
    
    INPUT
        t: scalar
            The hopping parameter of the system.
        tau: scalar
            The temprature of the hubbard dime to calc a set of kinetic 
            energies for.
        dvks: scalar or np.array
            The non-interacting site potnetial differnce or diffrences to 
            calc the kinetic energy for.
        
    OUTPUT
        Ts: scalar or np.array
    '''
    top = 4*t**2*np.tanh(np.sqrt(4*t**2+dvks**2)/(4*tau))
    bottom = np.sqrt(4*t**2+dvks**2)
    return -top/bottom
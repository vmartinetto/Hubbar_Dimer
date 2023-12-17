import numpy as np
import scipy as sp

def F(t,n1,n2):
    return -2*t*np.sqrt(n1*n2)

def Ts(t,n1,n2):
    return -2*t*np.sqrt(n1*n2)

def EH(U,n1,n2):
    return (U/2)*(n1**2+n2**2)

def EX(U,n1,n2):
    return -(U/4)*(n1**2+n2**2)

def EC(U,t,n1,n2):
    return F(t,n1,n2)-Ts(t,n1,n2)-EH(U,n1,n2)-EX(U,n1,n2)
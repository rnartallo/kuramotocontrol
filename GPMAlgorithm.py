from operator import matmul
import numpy as np
import cmath
def GPMAlgorithm(C):
    N = C.shape[0]
    x0 = np.zeros(N,dtype=complex)
    for i in range(0,N):
        x0[i] = cmath.exp(1j*np.random.rand())
    alpha = np.max([0, -(np.min(np.linalg.eig(C)[0]).real)])
    I_n = np.identity(N)
    C_tilde = C + alpha*I_n
    T_prec = Tfunction(C_tilde,x0)
    T_succ = np.zeros(N)
    stop = False
    while not stop:
        T_succ = Tfunction(C_tilde,T_prec)
        if np.linalg.norm(T_succ-T_prec)<1e-10:
            stop = True
        T_prec=T_succ
    Phases = CalculatePhases(T_succ)
    return Phases

def Tfunction(C_tilde,x):
    N = C_tilde.shape[0]
    T = np.zeros(N,dtype=complex)
    Cx = np.matmul(C_tilde,x,dtype=complex)
    for i in range(0,N):
        if Cx[i]==0:
            T[i] = x[i]
        else:
            T[i] = Cx[i]/(np.abs(Cx[i]))
    return T

def CalculatePhases(T_succ):
    N = len(T_succ)
    phases = np.zeros(N)
    for i in range(0,N):
        phases[i] = cmath.phase(T_succ[i])
    return phases

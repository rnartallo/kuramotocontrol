import numpy as np
from scipy.integrate import solve_ivp

def KuramotoModel(t,theta,N,w,A):
    dOsc = np.empty(N,1)
    for n in range(0,N):
        dOsc[i] = w[i]
        for i in range(0,N):
            dOsc[i] = dOsc[i]+A[n][i]*np.sin(theta[i]-theta[n])
    return dOsc

def SolveKuramotoModel(theta_0,T,A,w,N):
    sol =solve_ivp(KuramotoModel,[0,T],theta_0,args=(N,w,A))
    return sol



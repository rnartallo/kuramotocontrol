import numpy as np
from scipy.integrate import odeint

def KuramotoModel(theta,t,p):
    N,w,A = p
    dOsc = np.zeros(N)
    for n in range(0,N):
        dOsc[n] = w[n]
        for i in range(0,N):
            dOsc[n] = dOsc[n]+A[n][i]*np.sin(theta[i]-theta[n])
    return dOsc

def SolveKuramotoModel(theta_0,T,N,w,A,num_points):
    p=[N,w,A]
    t = np.linspace(0,T,num_points)
    sol =odeint(KuramotoModel,theta_0,t,args=(p,))
    return [sol,t]

def CalculatePhaseDiffs(oscs_end_values):
    x_acheived = []
    for i in range(0,len(oscs_end_values)-1):
        x_acheived.append(oscs_end_values[i+1]-oscs_end_values[i])
    return x_acheived
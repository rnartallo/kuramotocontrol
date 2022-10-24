from cmath import phase
import numpy as np
from scipy.integrate import odeint
from scipy.signal import butter,lfilter
import cmath

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
    for i in range(0,len(oscs_end_values)):
        x_acheived.append(oscs_end_values[i]-oscs_end_values[0])
    return x_acheived
 
def CalculateCorrelationMatrixFromPhases(phases):
    N = len(phases)
    FC = np.zeros((N,N))
    for i in range(0,N):
        for j in range(0,N):
            FC[i,j]=np.cos(phases[j]-phases[i])
    return FC

def CalculateCorrelationMatrix(x_min):
    N=len(x_min)
    corr = np.zeros((N,N))
    for i in range(0,N):
        for j in range(i+1,N):
            corr[i,j] = np.cos(x_min[j]-x_min[i])
    corr = corr + np.transpose(corr) + np.identity(N)
    return corr

def CalculateEdgeSet(A):
    edges = []
    for i in range(0,A.shape[0]):
        for j in range(0,A.shape[1]):
            if A[i][j]>0:
                edge = [min(i,j),max(i,j)]
                if edge not in edges:
                    edges.append(edge)
    return edges
            

def CalculateIncidenceMatrix(edges,N):
    B = np.zeros((N,len(edges)))
    for i in range(0,len(edges)):
        edge = edges[i]
        B[edge[0]][i]=-1
        B[edge[1]][i]=1
    return B

def CalculateSinMatrix(edges,x_desired_vector):
    D = np.zeros((len(edges),len(edges)))
    for i in range(0,len(edges)):
        D[i][i]=np.sin(x_desired_vector[i])
    return D

def CalculatePhaseDifferencesFromMin(x_min,edges):
    x_desired = np.zeros(len(edges))
    for e in range(0,len(edges)):
        edge = edges[e]
        x_desired[e] = x_min[edge[1]]-x_min[edge[0]]
    return x_desired


def CalculateDeltaFromAdj(A,edges):
    delta = np.zeros((len(edges),1))
    for e in range(0,len(edges)):
        delta[e] = A[edges[e][0],edges[e][1]]
    return delta

def CalculateAdjFromDelta(delta,edges,N):
    A = np.zeros((N,N))
    for e in range(0,len(edges)):
        edge = edges[e]
        A[edge[0]][edge[1]]=delta[e]
        A[edge[1]][edge[0]]=delta[e]
    return A


def butter_bandpass(lowcut, highcut, fs, order=2):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(x, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    z = lfilter(b, a, x,axis=0)
    return z

def CalculateCFromMeanPhaseDiff(phase_diff):
    N = phase_diff.shape[0]
    C = np.zeros((N,N),dtype=complex)+np.identity(N)
    for i in range(0,N):
        for j in range(i+1,N):
            C[i,j] = cmath.exp(1j*phase_diff[i,j])
            C[j,i] = cmath.exp(-1j*phase_diff[i,j])
    return C

def CalculateXMinFromPhases(phases):
    xmin = np.zeros(len(phases))
    for i in range(0,len(phases)):
        xmin[i] = phases[i]-phases[0]
    return xmin

def RemoveBounds(phases):
    phases_no_mod = phases
    T = phases.shape[0]
    N = phases.shape[1]
    for j in range(0,N):
        for t in range(0,T-1):
            if np.abs(phases[t,j]-phases[t+1,j])>5:
                phases_no_mod[t+1:T,j] = phases_no_mod[t+1:T,j] -2*np.pi
    return phases_no_mod
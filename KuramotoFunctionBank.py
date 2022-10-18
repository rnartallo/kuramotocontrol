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

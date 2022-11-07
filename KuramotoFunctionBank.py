from cmath import phase
import numpy as np
from scipy.integrate import odeint
from scipy.signal import butter,lfilter
from scipy.sparse.csgraph import connected_components
import cmath

def KuramotoModel(theta,t,p):
    N,w,A = p
    dOsc = np.zeros(N)
    for n in range(0,N):
        dOsc[n] = w[n]
        for i in range(0,N):
            dOsc[n] = dOsc[n]+A[n][i]*np.sin(theta[i]-theta[n])
    return dOsc

def KuramotoModelTDFrequency(theta,t,p):
    N,w,A,numpoints,T = p
    dOsc = np.zeros(N)
    for n in range(0,N):
        dOsc[n] = w[n,min(int(np.floor(t)),39)]
        for i in range(0,N):
            dOsc[n] = dOsc[n]+A[n][i]*np.sin(theta[i]-theta[n])
    return dOsc


def KuramotoModelTDFrequencyAdj(theta,t,p):
    N,w,A,numpoints,T = p
    dOsc = np.zeros(N)
    for n in range(0,N):
        dOsc[n] = w[n,min(int(np.floor(t)),39)]
        for i in range(0,N):
            dOsc[n] = dOsc[n]+A[n][i][min(int(np.floor(t)),39)]*np.sin(theta[i]-theta[n])
    return dOsc

def SolveKuramotoModel(theta_0,T,N,w,A,num_points):
    p=[N,w,A]
    t = np.linspace(0,T,num_points)
    sol =odeint(KuramotoModel,theta_0,t,args=(p,))
    return [sol,t]

def SolveKuramotoModeTDFrequency(theta_0,T,N,w,A,num_points):
    p=[N,w,A,num_points,T]
    t = np.linspace(0,T,num_points)
    sol =odeint(KuramotoModelTDFrequency,theta_0,t,args=(p,))
    return [sol,t]

def SolveKuramotoModeTDFrequencyAdj(theta_0,T,N,w,A,num_points):
    p=[N,w,A,num_points,T]
    t = np.linspace(0,T,num_points)
    sol =odeint(KuramotoModelTDFrequencyAdj,theta_0,t,args=(p,))
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

def CalculatePhaseLockingValue(phases):
    N = phases.shape[1]
    window_length = phases.shape[0]
    PLV = np.zeros((N,N)) + np.identity(N)
    for i in range(0,N-1):
        for k in range(i+1,N):
            sum =0
            for t in range(0,window_length):
                sum+= cmath.exp(1j*(phases[t,i]-phases[t,k]))
            PLV[i,k] = float(np.abs(sum/window_length))
            PLV[k,i] = PLV[i,k]
    return PLV

def CalculatePhaseLockingIndex(PLV):
    N = PLV.shape[0]
    PL_index = 0
    for i in range(0,N-1):
        for j in range(i+1,N):
            PL_index+=PLV[i,j]
    return PL_index/(N*(N-1)/2)

def CalculateSignedEdgeSet(A):
    pos_edges = []
    neg_edges =[]
    for i in range(0,A.shape[0]):
        for j in range(0,A.shape[1]):
            if A[i][j]>0:
                pos_edge = [min(i,j),max(i,j)]
                if pos_edge not in pos_edges:
                    pos_edges.append(pos_edge)
            if A[i][j]<0:
                neg_edge = [min(i,j),max(i,j)]
                if neg_edge not in neg_edges:
                    neg_edges.append(neg_edge)
    return [pos_edges,neg_edges]

def CalculateAdjFromEdges(edges,N):
    A = np.zeros((N,N))
    for e in edges:
        A[e[0],e[1]]=1
        #A[e[1],e[0]]=1
    return A

def DFS(v,visited,path,A,N):
    path.append(v)
    for j in range(0,N):
        if A[v,j]>0:
            visited[j]=1
            path,visited = DFS(j,visited,path,A,N)
    return path,visited

def CalculateConnectedComponents(A):
    cc =[]
    cc_info = connected_components(A)
    for i in range(0,cc_info[0]):
        c =[]
        for j in range(0,len(cc_info[1])):
            if cc_info[1][j]==i:
                c.append(j)
        cc.append(c)
    return cc

 
def CheckForNegEdgesSuperNode(pos_cc,neg_edges):
    for cc in pos_cc:
        for e in neg_edges:
            if e[0] in cc and e[1] in cc:
                return True
    return False

def CalculateSuperNegAdj(pos_cc,neg_edges):
    M = len(pos_cc)
    A = np.zeros((M,M))
    for i in range(0,M):
        for j in range(0,M):
            for e in neg_edges:
                if e[0] in pos_cc[i] and e[1] in pos_cc[j]:
                    A[i,j]=1
                    A[j,i]=1
    return A


def ExtractLayersFromSuperNegGraph(A):
    N = np.shape(A)[0]
    layers = N*np.ones(N)
    layers[0]=0
    for v in range(0,N):
        for j in range(0,N):
            if A[v,j]>0:
                layers[j] = min(layers[v]+1,layers[j])
    return layers


def CheckBalanceFromLayers(layers,A):
    N = np.shape(A)[0]
    for j in range(0,N):
        for k in range(0,N):
            if layers[j]==layers[k] and A[j,k]>0:
                return False
    return True





def DetermineStructuralBalance(W):
    N= np.shape(W)[0]
    pos_edges, neg_edges = CalculateSignedEdgeSet(W)
    pos_adj = CalculateAdjFromEdges(pos_edges,N)
    pos_cc = CalculateConnectedComponents(pos_adj)
    if CheckForNegEdgesSuperNode(pos_cc,neg_edges):
        return False
    super_neg_adj = CalculateSuperNegAdj(pos_cc,neg_edges)
    layers = ExtractLayersFromSuperNegGraph(super_neg_adj)
    return CheckBalanceFromLayers(layers,super_neg_adj)

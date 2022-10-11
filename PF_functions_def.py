#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 16:49:24 2022

@author: alvarem
"""

import math
import numpy as np
import matplotlib.pyplot as plt 
import progressbar
from scipy import linalg as la
from scipy.sparse import identity
from scipy.sparse import rand
from scipy.sparse import diags
from scipy.sparse import triu
import copy
from sklearn.linear_model import LinearRegression
from scipy.stats import ortho_group
import time

#%%

"""

def PF_ou(N,T,l,z,lz,collection_input):
    
    [dim,dim_o,A,R1,R2,H,m0,C0]=collection_input
    
    # In the following lines we initialize the PF
    x0=np.random.multivariate_normal(m0,C0,N)
    J=T*(2**l)
    I=identity(dim).toarray()
    #print(l)
    tau=1./2**(l)
    L=la.expm(A*tau)
    ## We are going to need W to be symmetric! 
    W=(R1@R1)@(la.inv(A+A.T)@(L@(L.T)-I))
    for k in range(J):
        #in the following we propagate the particles using 
"""





#%%
def M(x0,b,A,Sig,fi,l,d,N,dim):
    # This is the function for the transition Kernel M(x,du): R^{d_x}->P(E_l)
    # ARGUMENTS: the argument of the Kenel x0 \in R^{d_x} (rank 1 dimesion dim=d_x array)
    # the drift and diffusion b, and Sig respectively (rank 1 and 2 numpy arrays of dim=d_x respectively)
    # the level of discretization l, the distance of resampling, the number of
    # particles N, and the dimension of the problem dim=d_x
    # OUTCOMES: x, an array of rank 3  2**l*d,N,dim that represents the path simuled
    # along the discretized time for a number of particles N.
    steps=int(2**l*d)
    dt=1./2**l
    x=np.zeros((steps+1,N,dim))
    x[0]=x0
    I=identity(dim).toarray()

    for t in range(steps):
        dW=np.random.multivariate_normal(np.zeros(dim),I,N)*np.sqrt(dt)
        x[t+1]=x[t]+b(x[t],A)*dt+ dW@(Sig(x[t],fi).T)
    return x




def b_ou(x,A):
    # Returns the drift "vector" evaluated at x
    # ARGUMENTS: x is a rank two array with dimensions where the first dimension 
    # corresponds to the number of particles and the second to the dimension
    # of the probelm. The second argument is A, which is a squared rank 2 array 
    # with dimension of the dimesion of the problem
    
    # OUTPUTS: A rank 2 array where the first dimension corresponds to the number
    # of particles and the second to the dimension of the system.
        
        mult=x@(A.T)
        #mult=np.array(mult)*10
        return mult

#dim=len(x.T)



def Sig_ou(x,fi):
    # Returns the Ornstein-Oulenbeck diffusion matrix 
        
        return fi

"""
x=np.array([[1,0],[0,1],[10,10]])



l=5
d=20

N=10
dim=10

x0=np.random.normal(1,1,dim)
np.random.seed(3)
comp_matrix = ortho_group.rvs(dim)
print(comp_matrix)
inv_mat=la.inv(comp_matrix)
S=diags(np.random.normal(1,0.1,dim),0).toarray()
fi=inv_mat@S@comp_matrix

B=diags(np.random.normal(-1,0.1,dim),0).toarray()*(2/3)
B=inv_mat@B@comp_matrix
#B=comp_matrix-comp_matrix.T  +B 
np.random.seed(3)
x3=M(x0,b_ou,B,Sig_ou,fi,l,d,N,dim)
"""
#%%
"""
l=5
d=1
N=20
T=10
dim=5
dim_o=dim
#x=M(x0,b_ou,Sig_ou,l,d,N,dim)
I=identity(dim).toarray()
I_o=identity(dim_o).toarray()
#R2=(identity(dim_o).toarray() + np.tri(dim_o,dim_o,1) - np.tri(dim_o,dim_o,-2))/20
R2=I
H=rand(dim_o,dim,density=0.75).toarray()/1e-1

x0=np.random.normal(1,0,dim)
np.random.seed(3)
comp_matrix = ortho_group.rvs(dim)
inv_mat=la.inv(comp_matrix)
S=diags(np.random.normal(1,0.001,dim),0).toarray()
fi=inv_mat@S@comp_matrix

B=diags(np.random.normal(-.1,0.001,dim),0).toarray()*(2/3)
A=inv_mat@B@comp_matrix

#A=b_ou(I,B).T
R1=Sig_ou(np.zeros(dim),fi)
x= M(x0,b_ou,A,Sig_ou,fi,l,d,N,dim)
#%%
print(x.shape)
times=np.array(range(int(2**l*d+1)))/2**l
#plt.plot(times,x2[:,7,0])
plt.plot(times,x[:,0,0])
"""
#%%


def gen_data(T,l,collection_input):
    [dim,dim_o,A,R1,R2,H,m0,C0]=collection_input
    J=T*(2**l)
    I=identity(dim).toarray()
    tau=2**(-l)
    L=la.expm(A*tau)
    ## We are going to need W to be symmetric! 
    W=(R1@R1)@(la.inv(A+A.T)@(L@(L.T)-I))
    W=(W+W.T)/2.
    C=tau*H
    V=(R2@R2)*tau

    v=np.zeros((J+1,dim,1))
    z=np.zeros((J+1,dim_o,1))
    #v[0]=np.random.multivariate_normal(m0,C0,(1)).T
    v[0]=np.random.multivariate_normal(m0,C0,(1)).T
    z[0]=np.zeros((dim_o,1))


    for j in range(J):
        ## truth
        v[j+1] = L@v[j] + np.random.multivariate_normal(np.zeros(dim),W,(1)).T
        ## observation
        z[j+1] = z[j] + C@v[j+1] + np.random.multivariate_normal(np.zeros(dim_o),V,(1)).T
        
    return([z,v])


def cut(T,lmax,l,v):
    ind = np.arange(T*2**l+1)
    rtau = 2**(lmax-l)
    w = v[ind*rtau]
    return(w)

def KBF(T,l,lmax,z,collection_input):
    
    [dim,dim_o,A,R1,R2,H,m0,C0]=collection_input
    J=T*(2**l)
    I=identity(dim).toarray()
    tau=2**(-l)
    L=la.expm(A*tau)
    W=(R1@R1)@(la.inv(A+A.T)@(L@(L.T)-I))
    W=(W+W.T)/2.
    
    ## C: dim_o*dim matrix
    C=tau*H
    V=(R2@R2)*tau
    
    z=cut(T,lmax,l,z)
    m=np.zeros((J+1,dim,1))
    c=np.zeros((J+1,dim,dim))
    m[0]=np.array([m0]).T
    c[0]=C0
    
    for j in range(J):
       
        ## prediction mean-dim*1 vector
        mhat=L@m[j]
        ## prediction covariance-dim*dim matrix
        chat=L@c[j]@(L.T)+W
        ## innovation-dim_o*1 vector
        d=(z[j+1]-z[j])-C@mhat
        ## Kalman gain-dim*dim_o vector
        K=(chat@(C.T))@la.inv(C@chat@(C.T)+V)
        ## update mean-dim*1 vector
        
        m[j+1]=mhat+K@d
        ## update covariance-dim*dim matrix
        c[j+1]=(I-K@C)@chat
    return([m,c])

"""
"""

#%%
"""
l=8
d=10
N=10
T=10
dim=5
dim_o=dim
#x=M(x0,b_ou,Sig_ou,l,d,N,dim)


#R2=(identity(dim_o).toarray() + np.tri(dim_o,dim_o,1) - np.tri(dim_o,dim_o,-2))/20
H=rand(dim_o,dim,density=0.75).toarray()/1e-1
I=identity(dim).toarray()
R2=I
np.random.seed(3)
comp_matrix = ortho_group.rvs(dim)
inv_mat=la.inv(comp_matrix)
#A=b_ou(I,B).T
S=diags(np.random.normal(1,0.1,dim),0).toarray()
fi=inv_mat@S@comp_matrix

B=diags(np.random.normal(-.1,0.1,dim),0).toarray()*(2/3)
A=inv_mat@B@comp_matrix
R1=Sig_ou(np.zeros(dim),fi)
np.random.seed(2)
C0=I*1e-3
m0=np.random.multivariate_normal(np.zeros(dim),I)
collection_input=[dim,dim_o,A,R1,R2,H,m0,C0]


[z,x_true]=gen_data(T,l,collection_input)
#z=np.reshape(z,z.shape[:2])
"""

#%%

"""
lmax=l
kbf=KBF(T,l,lmax,z,collection_input)
plt.plot(2**(-l)*np.array(range(T*2**l+1)),x_true[:,0,0],label="Signal")
plt.plot(2**(-l)*np.array(range(T*2**l+1)),kbf[0][:,0,0],label="KBF")


#%%
lmax=8
l=lmax
d=10
np.random.seed(11)
x_star=M(x0,b_ou,B,Sig_ou,fi,l,d,N,dim)



print(x_star[-1])
"""
#%%
#[m,c]=KBF(T,l,lmax,z,collection_input)
def ht(x,H, para=True):
    #This function takes as argument a rank 3 array where for each element(2 rank array)
    # x[i]  the function applies x[i]@H.T
    #ARGUMENTS: rank 3 array x, para=True computes the code with the einsum 
    #function from numpy. Otherwise the function is computed with a for
    #in the time discretization
    #OUTPUTS: rank 3 array h
    
    
    if para==True:
        h=np.einsum("ij,tkj->tki",H,x)
    else:
        h=np.zeros(x.shape)
        for i in range(len(x)):
            h[i]=x[i]@(H.T)
    return h
            
            
    


def G(z,x,ht,H,l,para=True):
    #This function emulates the Radon-Nykodim derivative of the Girsanov formula
    #ARGUMENTS: z are the observations(2 rank array) , x is the array of the 
    #particles (rank 3 array, with 1 dimension less in the time discretization), 
    #ht is the function that computes the h(x) and d is the distance in which we
    #compute the paths.
    #OUTPUT: logarithm of the weights    
    h=ht(x,H,para=para)
    delta_z=z[1:]-z[:-1]
    delta_l=1./2**l
    suma1=np.einsum("tnd,td->n",h,delta_z)

    suma2=-(1/2.)*delta_l*np.einsum("tnj,tnj->n",h,h)
    #print(suma1,suma2)
    log_w=suma1+suma2
   
    return log_w

#%%
#tests for G
"""
z=np.array([[1,0],[2,0],[3,0]])
H=np.array([[1,0],[0,1]])
x=np.array([[[1,0],[3,0]],[[2,0],[4,0]]])
l=1
#print(ht(x,H))
print(G(z,x,ht,H,l,para=True))


#%%
print(x.shape)
times=np.array(range(int(2**l*d+1)))/2**l
plt.plot(times,x0[:,7,0])
#plt.plot(times,x1[:,0,0])


#%%
#zg=z[:2**l*d+1]
zg=np.zeros((T*2**l+1,10))+1
#print((T*2**l+1,10))
#print(x.shape)
xg=np.zeros((T*2**l,10,10))
#xg=np.zeros((2560, 10, 10))
lik=G(zg,x[:-1],ht,H,l,para=True)

print(lik)
#
"""
#%%

def sr(W,N,x,dim):
    # This function does 2 things, given probability weights (normalized)
    # it uses systematic resampling, and constructs the set of resampled 
    # particles.
    # ARGUMENTS: W: Normalized weights with dimension N (number of particles)
    # x: rank 2 array of the positions of the N particles, its dimesion is
    # Nxdim, where dim is the dimension of the problem.
    # OUTPUTs: part: is a N dimentional array where its value in the ith position
    # represents the number of times that particle was sampled.
    # x_new: is the new set of resampled particles.
    
    Wsum=np.cumsum(W)
    #np.random.seed()
    U=np.random.uniform(0,1./N)
    part=np.zeros(N,dtype=int)
    part[0]=np.floor(N*(Wsum[0]-U)+1)
    k=part[0]
    #print(U,part[0])
    for i in range(1,N):
        j=np.floor(N*(Wsum[i]-U)+1)
        part[i]=j-k
        k=j
    x_new=np.zeros((N,dim))
    k=0
    for i in range(N):
        for j in range(part[i]):
            x_new[k]=x[i]
            k+=1
    return [part, x_new]

def multi_samp(W,N,x,dim): #from multinomial sampling
    # This function does 2 things, given probability weights (normalized)
    # it uses multinomial resampling, and constructs the set of resampled 
    # particles.
    # ARGUMENTS: W: Normalized weights with dimension N (number of particles)
    # x: rank 2 array of the positions of the N particles, its dimesion is
    # Nxdim, where dim is the dimension of the problem.
    # OUTPUTs: part: is a N dimentional array where its value in the ith position
    # represents the number of times that particle was sampled.
    # x_new: is the new set of resampled particles.
    
    part_samp=np.random.choice(N,size=N,p=W) #particles resampled 
    #print(part_samp)
    x_resamp=x[part_samp]
    return [part_samp+1,x_resamp] #here we add 1 bc it is par_lab are thought 
    # as python labels, meaning that they start with 0.


"""
#This function is not finished, it is supposed to store the original paths 
#and the resampled paths.
def sr(W,N,x_or,dim,d_steps):
    # This function does 2 things, given probability weights (normalized)
    # it uses systematic resampling, and constructs the set of resampled 
    # particles.
    # ARGUMENTS: W: Normalized weights with dimension N (number of particles)
    # x_or(from x original): rank 3 array of the positions of the N 
    # particles in the discretized interval from i*d_steps:(i+1)*d_steps,
    # its dimesion is Nxdim, where dim is the dimension of the problem.
    # OUTPUTs: part: is a N dimentional array where its value in the ith position
    # represents the number of times that particle was sampled.
    # x_new: is the new set of resampled particles.
    
    Wsum=np.cumsum(W)
    #np.random.seed()
    U=np.random.uniform(0,1./N)
    part=np.zeros(N,dtype=int)
    part[0]=np.floor(N*(Wsum[0]-U)+1)
    k=part[0]
    #print(U,part[0])
    for i in range(1,N):
        j=np.floor(N*(Wsum[i]-U)+1)
        part[i]=j-k
        k=j
    x_new=np.zeros((d_steps,N,dim))
    k=0
    for i in range(N):
        for j in range(part[i]):
            x_new[:,k]=x_or[:,i]
            k+=1
    return [part, x_new]
"""
        
def norm_logweights(lw,ax=0):
    # returns the normalized weights given the log of the normalized weights 
    #ARGUMENTS: lw is a rank 1 array of the weights 
    #OUTPUT: w a rank 1 array of the same dimesion of lw 
    m=np.max(lw,axis=ax,keepdims=True)
    wsum=np.sum(np.exp(lw-m),axis=ax,keepdims=True)
    w=np.exp(lw-m)/wsum
    return w

#%%
"""
N=10
W=np.array(range(N))
W=W/np.sum(W)
seed_val=3
x=np.random.multivariate_normal(np.zeros(dim),I,N)
print(W,x)
print(sr(W,N,seed_val,x,dim))
"""

#%%

def PF(T,z,lmax,x0,b_ou,A,Sig_ou,fi,ht,H,l,d,N,dim,resamp_coef,para=True):
    # The particle filter function is inspired in 
    # Bain, A., Crisan, D.: Fundamentals of Stochastic Filtering. Springer,
    # New York (2009).
    # ARGUMENTS: T: final time of the propagation, T>0 and preferably integer
    # z: observation process, its a rank two array with discretized observation
    # at the intervals 2^(-lmax)i, i \in {0,1,...,T2^lmax}. with dimension
    # (T2^{lmax}+1) X dim
    # lmax: level of discretization of the observations
    # x0: initial condition of the particle filter, rank 1 array of dimension
    # dim
    # b_out: function that represents the drift of the process (its specifications
    # is already in the document. A is the arguments taht takes
    # Sig_out: function that represents the diffusion of the process (its specifications
    # is already in the document. Its arguments are included in fi.
    # ht: function in the observation process (its specifications
    # is already in the document). Its arguments are included in H.
    # d: time span in which the resampling is computed. d must be a divisor of T.
    # N: number of particles, N \in naturals greater than 1
    # dim: dimension of the problem
    # para: key to wheter compute the paralelization or not.
    # OUTPUTS: x: is the rank 3 array with the resampled particles at times 
    # 2^{-l}*i, i \in {0,1,..., T*2^l}, its dimentions are (2**l*T+1,N,dim)
    # log_weights: logarithm of the weights at times i*d, for i \in {0,1,...,T/d}.
    # it is a rank 2 array with dimensions (int(T/d),N)
    # suma: its the computation of the particle filter for each dimension of the problem
    # its a rank 2 array with dimensions (int(T/d),dim)
    x=np.zeros((2**l*T+1,N,dim))
    z=cut(T,lmax,l,z)
    log_weights=np.zeros((int(T/d),N))
    x_pf=np.zeros((int(T/d),N,dim))                            
    x_new=x0
    x[0]=x0
    d_steps=int(d*2**l)
    for i in range(int(T/d)):
        x[i*d_steps:(i+1)*d_steps+1]=M(x_new,b_ou,A,Sig_ou,fi,l,d,N,dim)
        xi=x[i*d_steps:(i+1)*d_steps]
        zi=z[i*d_steps:(i+1)*d_steps+1]
        log_weights[i]=G(zi,xi,ht,H,l,para=True)
        weights=norm_logweights(log_weights[i],ax=0)
        #seed_val=i
        #print(weights.shape)
        x_pf[i]=xi[-1]
        ESS=1/np.sum(weights**2)
        if ESS<resamp_coef*N:
            #[part0,part1,x0_new,x1_new]=max_coup_sr(w0,w1,N,xi0[-1],xi1[-1],dim)
            x_new=multi_samp(weights,N,x_pf[i],dim)[1]
        else:
            x_new=x_pf[i]
       #x_new=sr(weights,N,x_pf[i],dim)[1]
    #Filter
    #spots=np.arange(d_steps,2**l*T+1,d_steps,dtype=int)
    #x_pf=x[spots]
    #weights=norm_logweights(log_weights,ax=1)
    #print(x_pf.shape,weights.shape)
    #suma=np.sum(x_pf[:,:,1]*weights,axis=1)
    
    return [x,log_weights,x_pf]

"""
#This function is not finished, it is supposed to store the original paths 
#and the resampled paths.

def PF(T,z,lmax,x0,b_ou,A,Sig_ou,fi,ht,H,l,d,N,dim,para=True):
    # The particle filter function is inspired in 
    # Bain, A., Crisan, D.: Fundamentals of Stochastic Filtering. Springer,
    # New York (2009).
    # ARGUMENTS: T: final time of the propagation, T>0 and preferably integer
    # z: observation process, its a rank two array with discretized observation
    # at the intervals 2^(-lmax)i, i \in {0,1,...,T2^lmax}. with dimension
    # (T2^{lmax}+1) X dim
    # lmax: level of discretization of the observations
    # x0: initial condition of the particle filter, rank 1 array of dimension
    # dim
    # b_out: function that represents the drift of the process (its specifications
    # is already in the document. A is the arguments taht takes
    # Sig_out: function that represents the diffusion of the process (its specifications
    # is already in the document. Its arguments are included in fi.
    # ht: function in the observation process (its specifications
    # is already in the document). Its arguments are included in H.
    # d: time span in which the resampling is computed. d must be a divisor of T.
    # N: number of particles, N \in naturals greater than 1
    # dim: dimension of the problem
    # para: key to wheter compute the paralelization or not.
    # OUTPUTS: x: is the rank 3 array with the resampled particles at times 
    # 2^{-l}*i, i \in {0,1,..., T*2^l}, its dimentions are (2**l*T+1,N,dim)
    # log_weights: logarithm of the weights at times i*d, for i \in {0,1,...,T/d}.
    # it is a rank 2 array with dimensions (int(T/d),N)
    # suma: its the computation of the particle filter for each dimension of the problem
    # its a rank 2 array with dimensions (int(T/d),dim)
    x=np.zeros((2**l*T+1,N,dim))
    z=cut(T,lmax,l,z)
    log_weights=np.zeros((int(T/d),N))                            
    x_new=x0
    x_resamp=np.zeros((2**l*T,N,dim))
    x[0]=x0
    d_steps=int(d*2**l)
    for i in range(int(T/d)):
        xi=M(x_new,b_ou,A,Sig_ou,fi,l,d,N,dim)
        x[i*d_steps:(i+1)*d_steps]=xi[:-1]
     
        zi=z[i*d_steps:(i+1)*d_steps+1]
        log_weights[i]=G(zi,xi[:-1],ht,H,l,para=True)
        weights=norm_logweights(log_weights[i],ax=0)
        #seed_val=i
        #print(weights.shape)
        x_resamp=sr(weights,N,xi[:-1],dim)[1]
    #Filter
    spots=np.arange(d_steps,2**l*T+1,d_steps,dtype=int)
    x_pf=x[spots]
    weights=norm_logweights(log_weights,ax=1)
    #print(x_pf.shape,weights.shape)
    suma=np.sum(x_pf[:,:,1]*weights,axis=1)
    
    return [x,log_weights,suma]
"""

      
#%%

"""      
l=5
d=1./2**4
N=5
T=10
dim=3
dim_o=dim
#x=M(x0,b_ou,Sig_ou,l,d,N,dim)
I=identity(dim).toarray()
I_o=identity(dim_o).toarray()
#R2=(identity(dim_o).toarray() + np.tri(dim_o,dim_o,1) - np.tri(dim_o,dim_o,-2))/20
R2=I
np.random.seed(0)
H=rand(dim_o,dim,density=0.75).toarray()/1e-2

x0=np.random.normal(1,0,dim)
np.random.seed(3)
comp_matrix = ortho_group.rvs(dim)
inv_mat=la.inv(comp_matrix)
S=diags(np.random.normal(1,0.001,dim),0).toarray()
fi=inv_mat@S@comp_matrix

B=diags(np.random.normal(-.1,0.001,dim),0).toarray()
A=inv_mat@B@comp_matrix

#A=b_ou(I,B).T
R1=Sig_ou(np.zeros(dim),fi)

np.random.seed(3)
C0=I*1e-6
m0=np.random.multivariate_normal(x0,C0)
collection_input=[dim,dim_o,A,R1,R2,H,m0,C0]


[z,x_true]=gen_data(T,l,collection_input)
z=np.reshape(z,z.shape[:2])


[x,log_weights,x_pf]= PF(T,z,l,x0,b_ou,A,Sig_ou,R1,ht,H,l,d,N,dim,para=False)
      """
#%%
"""
lmax=l
a=1
d_steps=int(d*2**l)
spots=np.arange(d_steps,2**l*T+1,d_steps,dtype=int)
z=np.reshape(z,(2**l*T+1,dim,1))
times=np.array(range(int(2**l*T+1)))/2**l



kbf=KBF(T,l,lmax,z,collection_input)
weights=norm_logweights(log_weights,ax=1)
print(x_pf[:,:,a].shape)
xmean=np.sum(weights*x_pf[:,:,a],axis=1)
#plt.plot(times,z[:,a,0])
#plt.plot(times,x_true[:,a,0],label="True signal")
plt.plot(spots/2**l,kbf[0][spots,a,0],label="KBF")
#plt.plot(2**(-l)*np.array(range(T*2**l+1)),kbf[0][:,0,0])
plt.plot(spots/2**l,xmean,label="PF")
#plt.plot(times,x[:,:,a])
#plt.plot(times,xmean,label="mean of the propagation")

plt.legend()

#%%
a=np.array([0,1,2])
print(a.shape)

#%%
print(log_weights)
w=norm_logweights(log_weights[-1],ax=0)
w_total=norm_logweights(log_weights,ax=1)
print(w_total)

print(w)
print(x[-1])
seed_val=0
[part, x_new]=sr(w,N,seed_val,x[-1],dim)
print([part, x_new])

"""
from numpy import *
#import cPickle, gzip
from matplotlib.pyplot import *
import scipy.linalg as linalg
import scipy
import csv
import igraph
import itertools
from scipy import signal
import random
import operator
import pickle
import collections
from copy import deepcopy
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
import matplotlib as plt
import math
import networkx as nx
from networkx.generators.classic import empty_graph, path_graph, complete_graph
from scipy.sparse import coo_matrix
import json
import louvain
import pandas as pd
import igraph as ig
from scipy.spatial import distance
from scipy.cluster.hierarchy import dendrogram,linkage
import community

#girvan-newman algo works best and gives dendro. Louvain puts all symptoms in one community. Fastgreedy (clauset-newman-moore) doesn't work too well.

def pcc(X, Y):
   ''' Compute Pearson Correlation Coefficient. '''
   # Normalise X and Y
   X -= X.mean(0)
   Y -= Y.mean(0)
   # Standardise X and Y
   X /= X.std(0)
   Y /= Y.std(0)
   # Compute mean product
   return np.mean(X*Y)


matplotlib.rc('xtick', labelsize=13) 
matplotlib.rc('ytick', labelsize=13)

#dna stuff

dna = json.loads(open('dna_all_patients.json').read())

th=np.ones(23) #this gets written over by percentile method below
frac= 0.16

ver=8 #bl
config = json.loads(open('bl_v0'+str(ver)+'.json').read())
config1 = json.loads(open('bl_v0'+str(ver)+'_control.json').read())
dat=[]
dat_con=[]
datbl=[]
current =2017


v=[2,22,24,9,11,13,15,18,4,20,6,26,28,30,32,34] #organized by domains
v_control=[9,20,22,7,11,13,16,2,18,4,24,26,28,30,32]
bs=[1,21,23,8,10,12,14,17,3,19,5,25,27,29,31,33]

subnodelist = ['gender', 'age','JOLO','SDM','SFT','HVLT','LNS','MOCA','SEADL','RBDQ','ESS','SCOPA-AUT','GDS','STAI','UPDRS1','UPDRS2','UPDRS3','T-UPDRS']
notation=['C1','C1','C2','C2','C2','C2','C2','C2','C3','C4','C4','C5','C6','C6','C7','C7','C7','C7']
#subnodelist=notation #number nodes by community notation
subnodelist_split=deepcopy(subnodelist)

subnodelist_split.append("rs11060180_CC")
subnodelist_split.append("rs11060180_CT")
subnodelist_split.append("rs11060180_TT")
subnodelist_split.append("rs6430538_CC")
subnodelist_split.append("rs6430538_CT")
subnodelist_split.append("rs6430538_TT")
subnodelist_split.append("rs823118_CC")
subnodelist_split.append("rs823118_CT")
subnodelist_split.append("rs823118_TT")
subnodelist_split.append("rs356181_CC")
subnodelist_split.append("rs356181_CT")
subnodelist_split.append("rs356181_TT")


subnodelist.append("rs11060180")
subnodelist.append("rs6430538")
subnodelist.append("rs823118")
subnodelist.append("rs356181")

dir=[0,1,1,-1,-1,1,1,1,-1,-1,1,1,1,1,1,1,1,1,1] #0 is for patient number

for i in range(len(config)):
     dat1=config[i]
     dat1=[dat1[x] for x in dat1.keys()]
     dat11=list([dat1[index] for index in bs])
     dat11.insert(0,dat1[16]) #patno
     dat11.insert(1,current-dat1[0]) #age
     dat11.insert(1,dat1[7]) #gender
     if dat1[16] in dna.keys():
        dat11.insert(19,dna[dat1[16]][subnodelist[18]]) #dna
        dat11.insert(20,dna[dat1[16]][subnodelist[19]]) #dna
        dat11.insert(21,dna[dat1[16]][subnodelist[20]]) #dna
        dat11.insert(22,dna[dat1[16]][subnodelist[21]]) #dna
        dat11=[float(x) for x in dat11]
        dat.append(dat11)

for i in range(len(config1)):
     dat1=config1[i]
     dat1=[dat1[x] for x in dat1.keys()]
     dat11=list([dat1[index] for index in v_control])
     dat11.insert(0,dat1[16]) #patno
     dat11.insert(1,current-dat1[0]) #age
     dat11.insert(1,dat1[5]) #gender
     dat11.insert(9,100) #insert SEADL for control
     dat11=[float(x) for x in dat11]
     dat_con.append(dat11)

for i in range(len(config)):
     dat1=config[i]
     dat1=[dat1[x] for x in dat1.keys()]
     dat11=list([dat1[index] for index in bs])
     dat11.insert(0,dat1[16]) #patno
     dat11.insert(1,current-dat1[0]) #age
     dat11.insert(1,dat1[7]) #gender
     if dat1[16] in dna.keys():
        dat11.insert(19,dna[dat1[16]][subnodelist[18]]) #dna
        dat11.insert(20,dna[dat1[16]][subnodelist[19]]) #dna
        dat11.insert(21,dna[dat1[16]][subnodelist[20]]) #dna
        dat11.insert(22,dna[dat1[16]][subnodelist[21]]) #dna
        dat11=[float(x) for x in dat11]
        datbl.append(dat11)
     

          
dat=np.array(dat)
datbl=np.array(datbl)
dat_con=np.array(dat_con)

n=shape(dat)[1]-1 + 4*2 #n symptoms not counting pat number plus the extra gene types for the 4 genes
m=shape(dat)[0]

mean=np.mean(datbl, axis=0)
std=np.std(datbl, axis=0)

Seadl_std=np.std(datbl[:,9])
mean[9]=np.mean(datbl[:,9])
std[9]=Seadl_std #SEADL from baseline values

th_p=50#84.1
th_=np.percentile(datbl, th_p, axis=0)
th=(th_-mean)/std

gene_th=[]
A=np.zeros((n,m))#random.randint(100, size=(n, n))
for i in range(1,n+1-4*2): #runs over all symptoms except pat no.
          if i >18:
             for s in range(3):
                gene_th.append(sum(dat[:,i]==s)/(m-sum(dat[:,i]==-1)))
          for k in range(m): #runs over all people
               if i==1 and dat[k][i] ==2: #connected only to male
                  A[i-1][k]=(1-(th_p/100))*(m-sum(dat[:,i]==0))/sum(dat[:,i]==2)    #standard deviation value corresponding to number of males in population 
                  continue
               if i>18:
                  s=int(dat[k][i])
                  if s==-1:
                     A[18+gen*3 + s][k]=0
                     continue
                  gen=i-19
                  vf=(1-(th_p/100))*(m-sum(dat[:,i]==-1))/sum(dat[:,i]==s)
                  A[18+gen*3 + s][k]= vf #give them a value corresponding to their
                  continue
               z=(dat[k][i]-mean[i])/std[i]
               if abs(z)>=th[i]: 
                  if (z*dir[i]>th[i]):
                     A[i-1][k]+=1
                  elif (z*dir[i]==th[i]):
                     A[i-1][k]=(((1-(th_p/100))*m)-sum(datbl[:,i]>th_[i]*dir[i]))/sum(datbl[:,i]==th_[i]*dir[i])

                        
      
random.seed(42)
year1=A
pat1=dat[:,0]

total=0
noofcomm=[]
commsize=[]
x=[]
membership=[]
commcolor=[]
commnodes=[]
vall=[]

B=np.zeros((n,n))
C=np.zeros((m,m))
D=np.hstack((B,A))
E=np.hstack((A.T,C))
F=np.vstack((D,E))

print (type(A))
G1= nx.Graph(F)

part = community.best_partition(G1)
val = [part.get(node) for node in G1.nodes()]
mod = community.modularity(part,G1)
vall.append(val)

print(val)
edges = G1.edges()
weights = [G1[u][v]['weight'] for u,v in edges]
cmap=cm.jet

X1, Y1 = nx.bipartite.sets(G1)
x.append(np.unique(val[:len(X1)]))
noofcomm.append(len(x[0]))
total+=len(x[0])
for i in range(len(x[0])):
            place=np.where(val[:len(X1)]==x[0][i])
            commsize.append(len(place))
            print (place)
            commnodes.append(list(subnodelist_split[j] for j in place[0]))
            #commcolor.append(G2.vs[place[0]][color])



ver=4
config = json.loads(open('bl_v0'+str(ver)+'.json').read())
config1 = json.loads(open('bl_v0'+str(ver)+'_control.json').read())
dat=[]
dat_con=[]
datbl=[]
current =2017

v=[2,22,24,9,11,13,15,18,4,20,6,26,28,30,32,34] #organized by domains
v_control=[9,20,22,7,11,13,16,2,18,4,24,26,28,30,32]
bs=[1,21,23,8,10,12,14,17,3,19,5,25,27,29,31,33]




dir=[0,1,1,-1,-1,1,1,1,-1,-1,1,1,1,1,1,1,1,1,1] #0 is for patient number

for i in range(len(config)):
     dat1=config[i]
     dat1=[dat1[x] for x in dat1.keys()]
     dat11=list([dat1[index] for index in v])
     dat11.insert(0,dat1[16]) #patno
     dat11.insert(1,current-dat1[0]) #age
     dat11.insert(1,dat1[7]) #gender
     if dat1[16] in dna.keys():
        dat11.insert(19,dna[dat1[16]][subnodelist[18]]) #dna
        dat11.insert(20,dna[dat1[16]][subnodelist[19]]) #dna
        dat11.insert(21,dna[dat1[16]][subnodelist[20]]) #dna
        dat11.insert(22,dna[dat1[16]][subnodelist[21]]) #dna
        dat11=[float(x) for x in dat11]
        dat.append(dat11)

for i in range(len(config1)):
     dat1=config1[i]
     dat1=[dat1[x] for x in dat1.keys()]
     dat11=list([dat1[index] for index in v_control])
     dat11.insert(0,dat1[16]) #patno
     dat11.insert(1,current-dat1[0]) #age
     dat11.insert(1,dat1[5]) #gender
     dat11.insert(9,100) #insert SEADL for control
     dat11=[float(x) for x in dat11]
     dat_con.append(dat11)

for i in range(len(config)):
     dat1=config[i]
     dat1=[dat1[x] for x in dat1.keys()]
     dat11=list([dat1[index] for index in bs])
     dat11.insert(0,dat1[16]) #patno
     dat11.insert(1,current-dat1[0]) #age
     dat11.insert(1,dat1[7]) #gender
     if dat1[16] in dna.keys():
        dat11.insert(19,dna[dat1[16]][subnodelist[18]]) #dna
        dat11.insert(20,dna[dat1[16]][subnodelist[19]]) #dna
        dat11.insert(21,dna[dat1[16]][subnodelist[20]]) #dna
        dat11.insert(22,dna[dat1[16]][subnodelist[21]]) #dna
        dat11=[float(x) for x in dat11]
        datbl.append(dat11)
     

          
dat=np.array(dat)
datbl=np.array(datbl)
dat_con=np.array(dat_con)

n=shape(dat)[1]-1 + 4*2 #n symptoms not counting pat number plus the extra gene types for the 4 genes
m=shape(dat)[0]

mean=np.mean(datbl, axis=0)
std=np.std(datbl, axis=0)

Seadl_std=np.std(datbl[:,9])
mean[9]=np.mean(datbl[:,9])
std[9]=Seadl_std #SEADL from baseline values

A=np.zeros((n,m))#random.randint(100, size=(n, n))
for i in range(1,n+1-4*2): #runs over all symptoms except pat no.
          for k in range(m): #runs over all people
               if i==1 and dat[k][i] ==2: #connected only to male
                  A[i-1][k]=(1-(th_p/100))*(m-sum(dat[:,i]==0))/sum(dat[:,i]==2)    #standard deviation value corresponding to number of males in population 
                  continue
               if i>18:
                  s=int(dat[k][i])
                  if s==-1:
                     A[18+gen*3 + s][k]=0
                     continue
                  gen=i-19
                  vf=(1-(th_p/100))*(m-sum(dat[:,i]==-1))/sum(dat[:,i]==s)
                  A[18+gen*3 + s][k]= vf #give them a value corresponding to their
                  continue
               z=(dat[k][i]-mean[i])/std[i]
               if abs(z)>=th[i]: 
                  if (z*dir[i]>th[i]):
                     A[i-1][k]+=1
                  elif (z*dir[i]==th[i]):
                     A[i-1][k]=(((1-(th_p/100))*m)-sum(datbl[:,i]>th_[i]*dir[i]))/sum(datbl[:,i]==th_[i]*dir[i])
                        

random.seed(42)
pat2=dat[:,0]
year2=A


B=np.zeros((n,n))
C=np.zeros((m,m))
D=np.hstack((B,A))
E=np.hstack((A.T,C))
F=np.vstack((D,E))

print (type(A))
G2= nx.Graph(F)

part = community.best_partition(G2)
val = [part.get(node) for node in G2.nodes()]
mod = community.modularity(part,G2)
vall.append(val)
print(val)

edges = G2.edges()
weights = [G2[u][v]['weight'] for u,v in edges]
cmap=cm.jet

X2, Y2 = nx.bipartite.sets(G2)
x.append(np.unique(val[:len(X2)]))
noofcomm.append(len(x[1]))
total+=len(x[1])
for i in range(len(x[1])):
            place=np.where(val[:len(X2)]==x[1][i])
            commsize.append(len(place))
            print (place)
            commnodes.append(list(subnodelist_split[j] for j in place[0]))
            #commcolor.append(G2.vs[place[0]][color])

      



ver=6
config = json.loads(open('bl_v0'+str(ver)+'.json').read())
config1 = json.loads(open('bl_v0'+str(ver)+'_control.json').read())
dat=[]
dat_con=[]
datbl=[]
current =2017

v=[2,22,24,9,11,13,15,18,4,20,6,26,28,30,32,34] #organized by domains
v_control=[9,20,22,7,11,13,16,2,18,4,24,26,28,30,32]
bs=[1,21,23,8,10,12,14,17,3,19,5,25,27,29,31,33]



dir=[0,1,1,-1,-1,1,1,1,-1,-1,1,1,1,1,1,1,1,1,1] #0 is for patient number

for i in range(len(config)):
     dat1=config[i]
     dat1=[dat1[x] for x in dat1.keys()]
     dat11=list([dat1[index] for index in v])
     dat11.insert(0,dat1[16]) #patno
     dat11.insert(1,current-dat1[0]) #age
     dat11.insert(1,dat1[7]) #gender
     if dat1[16] in dna.keys():
        dat11.insert(19,dna[dat1[16]][subnodelist[18]]) #dna
        dat11.insert(20,dna[dat1[16]][subnodelist[19]]) #dna
        dat11.insert(21,dna[dat1[16]][subnodelist[20]]) #dna
        dat11.insert(22,dna[dat1[16]][subnodelist[21]]) #dna
        dat11=[float(x) for x in dat11]
        dat.append(dat11)

for i in range(len(config1)):
     dat1=config1[i]
     dat1=[dat1[x] for x in dat1.keys()]
     dat11=list([dat1[index] for index in v_control])
     dat11.insert(0,dat1[16]) #patno
     dat11.insert(1,current-dat1[0]) #age
     dat11.insert(1,dat1[5]) #gender
     dat11.insert(9,100) #insert SEADL for control
     dat11=[float(x) for x in dat11]
     dat_con.append(dat11)

for i in range(len(config)):
     dat1=config[i]
     dat1=[dat1[x] for x in dat1.keys()]
     dat11=list([dat1[index] for index in bs])
     dat11.insert(0,dat1[16]) #patno
     dat11.insert(1,current-dat1[0]) #age
     dat11.insert(1,dat1[7]) #gender
     if dat1[16] in dna.keys():
        dat11.insert(19,dna[dat1[16]][subnodelist[18]]) #dna
        dat11.insert(20,dna[dat1[16]][subnodelist[19]]) #dna
        dat11.insert(21,dna[dat1[16]][subnodelist[20]]) #dna
        dat11.insert(22,dna[dat1[16]][subnodelist[21]]) #dna
        dat11=[float(x) for x in dat11]
        datbl.append(dat11)
     

          
dat=np.array(dat)
datbl=np.array(datbl)
dat_con=np.array(dat_con)

n=shape(dat)[1]-1 + 4*2 #n symptoms not counting pat number plus the extra gene types for the 4 genes
m=shape(dat)[0]

mean=np.mean(datbl, axis=0)
std=np.std(datbl, axis=0)

Seadl_std=np.std(datbl[:,9])
mean[9]=np.mean(datbl[:,9])
std[9]=Seadl_std #SEADL from baseline values

A=np.zeros((n,m))#random.randint(100, size=(n, n))
for i in range(1,n+1-4*2): #runs over all symptoms except pat no.
          for k in range(m): #runs over all people
               if i==1 and dat[k][i] ==2: #connected only to male
                  A[i-1][k]=(1-(th_p/100))*(m-sum(dat[:,i]==0))/sum(dat[:,i]==2)    #standard deviation value corresponding to number of males in population 
                  continue
               if i>18:
                  s=int(dat[k][i])
                  if s==-1:
                     A[18+gen*3 + s][k]=0
                     continue
                  gen=i-19
                  vf=(1-(th_p/100))*(m-sum(dat[:,i]==-1))/sum(dat[:,i]==s)
                  A[18+gen*3 + s][k]= vf #give them a value corresponding to their
                  continue
               z=(dat[k][i]-mean[i])/std[i]
               if abs(z)>=th[i]: 
                  if (z*dir[i]>th[i]):
                     A[i-1][k]+=1
                  elif (z*dir[i]==th[i]):
                     A[i-1][k]=(((1-(th_p/100))*m)-sum(datbl[:,i]>th_[i]*dir[i]))/sum(datbl[:,i]==th_[i]*dir[i])
                        

         
                    
random.seed(42)

year3=A
pat3=dat[:,0]


B=np.zeros((n,n))
C=np.zeros((m,m))
D=np.hstack((B,A))
E=np.hstack((A.T,C))
F=np.vstack((D,E))

print (type(A))
G3= nx.Graph(F)

part = community.best_partition(G3)
val = [part.get(node) for node in G3.nodes()]
mod = community.modularity(part,G3)
vall.append(val)


edges = G3.edges()
weights = [G3[u][v]['weight'] for u,v in edges]
cmap=cm.jet

X3, Y3 = nx.bipartite.sets(G3)
x.append(np.unique(val[:len(X3)]))
noofcomm.append(len(x[2]))
total+=len(x[2])
for i in range(len(x[2])):
            place=np.where(val[:len(X3)]==x[2][i])
            commsize.append(len(place))
            print (place)
            commnodes.append(list(subnodelist_split[j] for j in place[0]))
            #commcolor.append(G2.vs[place[0]][color])



B=np.zeros((n,n))
C=np.zeros((m,m))
D=np.hstack((B,A))
E=np.hstack((A.T,C))
F=np.vstack((D,E))



ver=8
config = json.loads(open('bl_v0'+str(ver)+'.json').read())
config1 = json.loads(open('bl_v0'+str(ver)+'_control.json').read())
dat=[]
dat_con=[]
datbl=[]
current =2017

v=[2,22,24,9,11,13,15,18,4,20,6,26,28,30,32,34] #organized by domains
v_control=[9,20,22,7,11,13,16,2,18,4,24,26,28,30,32]
bs=[1,21,23,8,10,12,14,17,3,19,5,25,27,29,31,33]



dir=[0,1,1,-1,-1,1,1,1,-1,-1,1,1,1,1,1,1,1,1,1] #0 is for patient number

for i in range(len(config)):
     dat1=config[i]
     dat1=[dat1[x] for x in dat1.keys()]
     dat11=list([dat1[index] for index in v])
     dat11.insert(0,dat1[16]) #patno
     dat11.insert(1,current-dat1[0]) #age
     dat11.insert(1,dat1[7]) #gender
     if dat1[16] in dna.keys():
        dat11.insert(19,dna[dat1[16]][subnodelist[18]]) #dna
        dat11.insert(20,dna[dat1[16]][subnodelist[19]]) #dna
        dat11.insert(21,dna[dat1[16]][subnodelist[20]]) #dna
        dat11.insert(22,dna[dat1[16]][subnodelist[21]]) #dna
        dat11=[float(x) for x in dat11]
        dat.append(dat11)

for i in range(len(config1)):
     dat1=config1[i]
     dat1=[dat1[x] for x in dat1.keys()]
     dat11=list([dat1[index] for index in v_control])
     dat11.insert(0,dat1[16]) #patno
     dat11.insert(1,current-dat1[0]) #age
     dat11.insert(1,dat1[5]) #gender
     dat11.insert(9,100) #insert SEADL for control
     dat11=[float(x) for x in dat11]
     dat_con.append(dat11)

for i in range(len(config)):
     dat1=config[i]
     dat1=[dat1[x] for x in dat1.keys()]
     dat11=list([dat1[index] for index in bs])
     dat11.insert(0,dat1[16]) #patno
     dat11.insert(1,current-dat1[0]) #age
     dat11.insert(1,dat1[7]) #gender
     if dat1[16] in dna.keys():
        dat11.insert(19,dna[dat1[16]][subnodelist[18]]) #dna
        dat11.insert(20,dna[dat1[16]][subnodelist[19]]) #dna
        dat11.insert(21,dna[dat1[16]][subnodelist[20]]) #dna
        dat11.insert(22,dna[dat1[16]][subnodelist[21]]) #dna
        dat11=[float(x) for x in dat11]
        datbl.append(dat11)
     

          
dat=np.array(dat)
datbl=np.array(datbl)
dat_con=np.array(dat_con)

n=shape(dat)[1]-1 + 4*2 #n symptoms not counting pat number plus the extra gene types for the 4 genes
m=shape(dat)[0]

mean=np.mean(datbl, axis=0)
std=np.std(datbl, axis=0)

Seadl_std=np.std(datbl[:,9])
mean[9]=np.mean(datbl[:,9])
std[9]=Seadl_std #SEADL from baseline values

A=np.zeros((n,m))#random.randint(100, size=(n, n))
for i in range(1,n+1-4*2): #runs over all symptoms except pat no.
          for k in range(m): #runs over all people
               if i==1 and dat[k][i] ==2: #connected only to male
                  A[i-1][k]=(1-(th_p/100))*(m-sum(dat[:,i]==0))/sum(dat[:,i]==2)    #standard deviation value corresponding to number of males in population 
                  continue
               if i>18:
                  s=int(dat[k][i])
                  if s==-1:
                     A[18+gen*3 + s][k]=0
                     continue
                  gen=i-19
                  vf=(1-(th_p/100))*(m-sum(dat[:,i]==-1))/sum(dat[:,i]==s)
                  A[18+gen*3 + s][k]= vf #give them a value corresponding to their
                  continue
               z=(dat[k][i]-mean[i])/std[i]
               if abs(z)>=th[i]: 
                  if (z*dir[i]>th[i]):
                     A[i-1][k]+=1
                  elif (z*dir[i]==th[i]):
                     A[i-1][k]=(((1-(th_p/100))*m)-sum(datbl[:,i]>th_[i]*dir[i]))/sum(datbl[:,i]==th_[i]*dir[i])
                        
               

random.seed(42)

year4=A
pat4=dat[:,0]



index = np.argwhere(pat1==3001) #delete this one person to make traj and comm space people numbers in all 4 years match for hybrid model.
pat1 = np.delete(pat1, index)
pat2 = np.delete(pat2, index)
pat3 = np.delete(pat3, index)
pat4 = np.delete(pat4, index)


p=[pat1,pat2,pat3,pat4]
pat_set=set(p[0]).intersection(*p)
#pat_set=pat_set.remove(3001.0)
pat_num=list(pat_set)

one=[i for i, val in enumerate(pat1) if val in pat_set]
two=[i for i, val in enumerate(pat2) if val in pat_set]
three=[i for i, val in enumerate(pat3) if val in pat_set]
four=[i for i, val in enumerate(pat4) if val in pat_set]

year1=year1[:,one]
year2=year2[:,two]
year3=year3[:,three]
year4=year4[:,four]

F=[]
F.append(year1)
F.append(year2)
F.append(year3)
F.append(year4)
F=np.array(F)

print (shape(F), 'F')
shap=shape(F)
for i in range(shap[0]):
   for j in range(shap[1]):
       if j>0 and j<18:
          for k in range(shap[2]):
             if F[i,j,k]>0:
                F[i,j,k]=1



print (F[:,:,0], 'stop')
c=198
F_=deepcopy(F)
F_[F_>0]=1
test=F_[:,:,c:]
F_=F_[:,:,:c]
test_pat_num=pat_num[c:]
pat_num=pat_num[:c]

num_p=len(pat_num)
M=np.zeros((num_p,num_p))


#first order
'''for i in range(num_p):
   for j in range(num_p):
      
      M[i][j]=(F[:,:,i]==F[:,:,j]).sum()'''


#second order
#pat_traj matrix. All trajectories overlapping with this patients traj are found through taking element wise product of all patient traj with this pat. Then if the total number of ones are greater than some threshold (strong similarity) then they pass through this patient's traj.
#Now we connect trajectories. Two patients/traj are connected based on combintaions of all the other trajectories passing through them. fraction of common trajectory edges in all combinations of traj/ union of all traject (with repetition included to account for stronger links if edge is repeated multiple times in different trajectories arising from patient i and j.)


'''x=[]
for i in range(num_p):
   x.append([])
   for j in range(num_p):
      F_one=F_[:,:,i]
      F_two=F_[:,:,j]
      F_shape=np.zeros((shape(F_)[0],shape(F_)[1]))
      for ss in range(shape(F_)[0]):
         for tt in range(shape(F_)[1]):
            if F_[ss,tt,i]==F_[ss,tt,j]:
               F_shape[ss,tt]=1
      x[i].append(F_shape) # all traj matching with traj of node i all traj with 75percentile of the traj coinciding with our traj
   zer=[]
   for j in range(len(x[i])):
      zer.append(np.count_nonzero(x[i][j]))

   thresh=np.percentile(zer, 90)
   b=list(np.where(array(zer)>thresh))[0]
   x[i]=[x[i][v] for v in b]


for i in range(num_p):
   for j in range(num_p):
      total=0
      c=[]
      yo11=[]
      yo22=[]
      for g in range(len(x[i])):
         d=x[i][g].flatten()
         dd=np.where(array(d)==1)[0]
         yo1=[n for n in dd]
         yo11.append(yo1)
      yo11=[val for sublist in yo11 for  val in sublist]
      for h in range(len(x[j])):
         e=x[j][h].flatten()
         ee=np.where(array(e)==1)[0]
         yo2=[n for n in ee]
         yo22.append(yo2)
      yo22=[val for sublist in yo22 for  val in sublist]

            #c.append(np.multiply(x[i][g], x[j][h]))

      for a in x[i]:
         total+=np.count_nonzero(a)
      for b in x[j]:
         total+=np.count_nonzero(b)
      rep=[]   
      for elem in deepcopy(yo11):
         if elem in yo22:
            yo11.pop(yo11.index(elem))
            rep.append(yo22.pop(yo22.index(elem)))
      if i==j:
         print (2*len(rep),total,i,'fd')
      if total==0:
         continue
      M[i][j]=len(rep)*2/total
   
      

np.savetxt('traj_in_variablespace_dot_product_second_order_th_half_baseline_train_.txt', M)'''
M=np.loadtxt('traj_in_variablespace_dot_product_second_order_th_half_baseline_train_.txt')



H=nx.Graph(M)              
part = community.best_partition(H)
values = [part.get(node) for node in H.nodes()]
mod = community.modularity(part,H)

print (mod, 'mod')
nocomm=len(np.unique(values))
pat_comm = dict(zip(pat_num, values))


#print ('Patients classified into communities based on dot product of variable trajectory: ' , pat_comm)
print (values)

# plot patient profile in each community:
uni=np.unique(values)
no_of_ppl=[]

prof=[]
prost=[]
for i in range(len(uni)):
   prof.append([])
   zorro=np.where(values==uni[i])[0]
   for s in range(len(zorro)):
      prof[i].append(F_[:,:,zorro[s]])  #genes are binarized
   joll=np.std(prof[i],axis=0)
   prost.append(joll)
   no_of_ppl.append(shape(prof[i])[0])
   prof[i]=np.mean(prof[i],axis=0)
   

print (no_of_ppl, ':number of ppl in each comm')
print (len(no_of_ppl), ':number of communities')

rem=[]
for i in range(len(no_of_ppl)):
   if no_of_ppl[i]<=10:
      rem.append(i)
prof=np.delete(prof, rem, axis=0)


a=np.mean(F_,axis=2)  #population average genes are not binarized for plotting- baseline all colors look the same






'''test_ppl=np.shape(test)[2]   #number of test people
comm=shape(prof)[0] #number of communtiies
predict_comm=np.zeros((test_ppl,comm))
for i in range(test_ppl):
   for j in range(comm):
      s=np.linalg.norm(test[0,:,i]-prof[j,0,:])
      predict_comm[i,j]=s

predictions=np.argmax(predict_comm,axis=1)   # predictions based on baseline year


correct=[]
world=[]
score=np.zeros((test_ppl,comm))
for i in range(test_ppl):
   j=predictions[i]
   s=np.linalg.norm(test[:,:,i]-prof[j,:,:])
   score[i,j]=s
   correct.append(s)
   s_=np.linalg.norm(test[:,:,i]-a)
   world.append(s_)


figure()
scatter(correct,world, color='r')
xlabel('Dist(pat_profile, pred_comm_profile)',fontsize=18)
xlabel('Dist(pat_profile, popultn_profile)',fontsize=18)
bbox_layout='tight'
show()'''






prof1=np.array(deepcopy(prof))
                        

pcnt=np.count_nonzero(F_, axis=2)[0]/shape(F_)[2] #consider only the baseline

prof1[prof1<=pcnt]=0
prof1[prof1>pcnt]=1      #binarize community profiles


test_ppl=np.shape(test)[2]   #number of test people
comm=shape(prof1)[0] #number of communities
predict_comm=np.zeros((test_ppl,comm))
for i in range(test_ppl):
   for j in range(comm):
      s=np.linalg.norm(test[0,:,i]-prof1[j,0,:])
      predict_comm[i,j]=s

predictions=np.argmin(predict_comm,axis=1)


'''order=[]
for i in range(test_ppl):
   s=np.linalg.norm(test[0,:,i]-prof1[predictions[i],0,:])
   order.append(s)
   
order_ppl=np.argsort(order)   #ordering ppl from least to most similarity with their predicted community
predictions=predictions[order_ppl]'''




############calculate distance of test patients from all others

correct=[]
wrong=[]
score=np.zeros((test_ppl,comm))
for i in range(test_ppl):
   wrong.append([])
   for j in range(comm):
      s=np.linalg.norm(test[:,:,order_ppl[i]]-prof1[j,:,:])
      score[i,j]=s
      if j==predictions[i]:
         correct.append(s)
      else:
         wrong[i].append(s)

wrong=np.array(wrong)
actual=np.argmin(score, axis=1)



figure()
scatter(np.arange(test_ppl),correct, color='r')
for i in range(shape(wrong)[1]):
   scatter(np.arange(test_ppl),wrong.T[i], color='b')
show()

frac_corr=np.count_nonzero(actual==predictions)/test_ppl
print (frac_corr, 'fraction correct')


##############################

correct=np.zeros((4,test_ppl))
world=np.zeros((comm,4,test_ppl))



for i in range(test_ppl):
   for j in range(4):
      for k in range(comm):
         s=np.linalg.norm(test[j,:,order_ppl[i]]-prof1[k,j,:])
         world[k,j,i]=s
         if k==predictions[i]:
            correct[j,i]=s
      

markers = ["." , "," , "o" , "v" , "^" , "<", ">"]
figure()
for i in range(4):
   subplot(4,1,i+1)
   for j in range(comm):
      corre=np.where(np.ones(test_ppl)*j==predictions)[0]
      col=['k']*test_ppl
      for s in range(len(corre)):
         col[corre[s]]='r'
      scatter(np.arange(test_ppl),world[j,i,:], color=np.array(col),marker=markers[j],label='dist_w_comm'+str(j))
      ylabel('Dist in \n year' + str(i), fontsize=12)
xlabel('Test-patients_decreasingly_confident_baseline_predictions',fontsize=14)
legend(loc=4)
tight_layout()
savefig('predictions_%.2f_50pc.jpeg'%frac_corr)

no_of_ppl=np.delete(no_of_ppl, rem, axis=0)

#a[0][18:]=1
figure()
fig, axes = subplots(nrows=comm, ncols=1) #if you need to plot cbar, plot here and then crop out common colorbar
i=0
for ax in axes.flat:
    ax.set_yticks(range(0,4))
    im = ax.imshow(prof[i]/a[0],cmap='Greys')
    ax.set_ylabel('Years \n'+str(no_of_ppl[i])+ ' ppl',fontsize=16)
    i+=1
    if i!=comm:
       ax.set_xticks([])
    else:
       ax.set_xticks(np.arange(len(subnodelist_split)))
       ax.set_xticklabels(subnodelist_split, rotation=90)

xlabel('Variables',fontsize=16)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
savefig('profile_50pcbinarization.jpg', bbox_inches='tight', minor=True)


gene_bar=[]
for i in range(shape(prof)[0]):   #across the communities
   s=prof[i,0,18:]*no_of_ppl[i]
   for j in range(int(len(s)/3.0)):   #across the genes
      s1=list(s[j*3:j*3+3])
      s1.append(no_of_ppl[i]-sum(s1))   #appending the NA gene
      gene_bar.append(s1)




y_pos = np.arange(j+1)
bar(y_pos,gene_bar[0],align='center')
show()

      
gene_split=['XX','xx','Xx','NA']
figure()
fig, axes = subplots(nrows=comm, ncols=1) #if you need to plot cbar, plot here and then crop out common colorbar
i=0
gene_bar=np.array(gene_bar)
for ax in axes.flat:
    im = ax.bar(y_pos,gene_bar[i*4+0],width=0.2,color='b',align='center', label='G1')
    im = ax.bar(y_pos-0.2,gene_bar[i*4+1],width=0.2,color='g',align='center', label='G2')
    im = ax.bar(y_pos+0.2,gene_bar[i*4+2],width=0.2,color='r',align='center', label='G3')
    im = ax.bar(y_pos+0.4,gene_bar[i*4+3],width=0.2,color='k',align='center', label='G4')
    ax.set_ylabel('Gene popltn \n'+str(no_of_ppl[i])+ ' ppl',fontsize=10)
    #ax.set_ylabel('Gene_population',fontsize=16)
    if i==0:
       ax.legend()
    i+=1
    if i!=comm:
       ax.set_xticks([])
    else:
       ax.set_xticks(np.arange(len(gene_split)))
       ax.set_xticklabels(gene_split)

xlabel('Gene_variants',fontsize=16)
fig.subplots_adjust(right=0.8)
savefig('gene_distribution_50pcbinarization.jpg', bbox_inches='tight', minor=True)


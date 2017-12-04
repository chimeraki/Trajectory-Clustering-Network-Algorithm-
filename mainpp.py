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
from copy import deepcopy
import operator
import pickle
import collections
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

th =0.5 #std threshold

ver=8 #bl
config = json.loads(open('bl_v0'+str(ver)+'.json').read())
config1 = json.loads(open('bl_v0'+str(ver)+'_control.json').read())
dat=[]
dat_con=[]
datbl=[]
m=len(config)#-2 #since we're getting rid of one guy who is in his own community for mean std bl . Patient no. 3211 and 3078

current =2017

v=[2,22,24,9,11,13,15,18,4,20,6,26,28,30,32,34] #organized by domains
v_control=[9,20,22,7,11,13,16,2,18,4,24,26,28,30,32]
bs=[1,21,23,8,10,12,14,17,3,19,5,25,27,29,31,33]

subnodelist = ['gender', 'age','JOLO','SDM','SFT','HVLT','LNS','MOCA','SEADL','RBDQ','ESS','SCOPA-AUT','GDS','STAI','UPDRS1','UPDRS2','UPDRS3','T-UPDRS']
notation=['C1','C1','C2','C2','C2','C2','C2','C2','C3','C4','C4','C5','C6','C6','C7','C7','C7','C7']
#subnodelist=notation #number nodes by community notation

dir=[0,1,1,-1,-1,1,1,1,-1,-1,1,1,1,1,1,1,1,1,1] #0 is for patient number

for i in range(len(config)):
     dat1=config[i]
     dat1=[dat1[x] for x in dat1.keys()]
     dat11=list([dat1[index] for index in bs])
     dat11.insert(0,dat1[16]) #patno
     dat11.insert(1,current-dat1[0]) #age
     dat11.insert(1,dat1[7]) #gender
     dat11=[float(x) for x in dat11]
     dat.append(dat11)

for i in range(len(config)):
     dat1=config[i]
     dat1=[dat1[x] for x in dat1.keys()]
     dat11=list([dat1[index] for index in bs])
     dat11.insert(0,dat1[16]) #patno
     dat11.insert(1,current-dat1[0]) #age
     dat11.insert(1,dat1[7]) #gender
     dat11=[float(x) for x in dat11]
     datbl.append(dat11)
     
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


          
dat=np.array(dat)
datbl=np.array(datbl)
dat_con=np.array(dat_con)
print (shape(dat))
#dat=dat[dat[:,0]!=3078,:]
#dat=dat[dat[:,0]!=3211,:]#remove this guy because he's in his own community with no symptoms, hence he has no trajectory
n=len(dat11)-1 #n symptoms not counting pat number
print (n)

mean=np.mean(dat_con, axis=0)
std=np.std(dat_con, axis=0)

Seadl_std=np.std(datbl, axis=0)
std[9]=Seadl_std[9] #SEADL from baseline values

A=np.zeros((n,m))#random.randint(100, size=(n, n))
for i in range(1,n+1): #runs over all symptoms except pat no.
          for k in range(m): #runs over all people
               if i==1:
                  A[i-1][k]=abs(dat[k][i]) #connected only to male
               z=(dat[k][i]-mean[i])/std[i] 
               if abs(z)>=th:
                  if i ==1: #gender
                       continue
                  elif (z*dir[i]>=th):
                       A[i-1][k]+=abs(z)
               
                    
random.seed(42)
year1=A
pat1=dat[:,0]
#index = np.argwhere(pat1==3078) 
#pat1 = np.delete(pat1, index)
#index = np.argwhere(pat1==3211) 
#pat1 = np.delete(pat1, index)



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
print(val, 'val')
mod = community.modularity(part,G1)
vall.append(val)
print(val)

edges = G1.edges()
weights = [G1[u][v]['weight'] for u,v in edges]
cmap=cm.jet

X1, Y1 = nx.bipartite.sets(G1)
print(X1, Y1, 'kk')
x.append(np.unique(val[:len(X1)]))
noofcomm.append(len(x[0]))
print(val[:len(X1)], len(X1),noofcomm,'jj')
total+=len(x[0])
for i in range(len(x[0])):
            place=np.where(val[:len(X1)]==x[0][i])
            print (place)
            commnodes.append(list(subnodelist[j] for j in place[0]))
            #commcolor.append(G2.vs[place[0]][color])



ver=4
config = json.loads(open('bl_v0'+str(ver)+'.json').read())
config1 = json.loads(open('bl_v0'+str(ver)+'_control.json').read())
dat=[]
dat_con=[]
datbl=[]
m=len(config)



for i in range(len(config)):
     dat1=config[i]
     dat1=[dat1[x] for x in dat1.keys()]
     dat11=list([dat1[index] for index in v])
     dat11.insert(0,dat1[16]) #patno
     dat11.insert(1,current-dat1[0]) #age
     dat11.insert(1,dat1[7]) #gender
     dat11=[float(x) for x in dat11]
     dat.append(dat11)

for i in range(len(config)):
     dat1=config[i]
     dat1=[dat1[x] for x in dat1.keys()]
     dat11=list([dat1[index] for index in bs])
     dat11.insert(0,dat1[16]) #patno
     dat11.insert(1,current-dat1[0]) #age
     dat11.insert(1,dat1[7]) #gender
     dat11=[float(x) for x in dat11]
     datbl.append(dat11)
     
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


          
dat=np.array(dat)
datbl=np.array(datbl)
dat_con=np.array(dat_con)

mean=np.mean(dat_con, axis=0)
std=np.std(dat_con, axis=0)

Seadl_std=np.std(datbl, axis=0)
std[9]=Seadl_std[9] #SEADL from baseline values


A=np.zeros((n,m))#random.randint(100, size=(n, n))
for i in range(1,n+1): #runs over all symptoms except pat no.
          for k in range(m): #runs over all people
               if i==1:
                  A[i-1][k]=abs(dat[k][i]) #connected only to male
               z=(dat[k][i]-mean[i])/std[i] 
               if abs(z)>=th:
                  if i ==1: #gender
                       continue
                  elif (z*dir[i]>=th):
                       A[i-1][k]+=abs(z)
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
            print (place)
            commnodes.append(list(subnodelist[j] for j in place[0]))
            #commcolor.append(G2.vs[place[0]][color])

      



ver=6
config = json.loads(open('bl_v0'+str(ver)+'.json').read())
config1 = json.loads(open('bl_v0'+str(ver)+'_control.json').read())
dat=[]
dat_con=[]
datbl=[]
m=len(config)



for i in range(len(config)):
     dat1=config[i]
     dat1=[dat1[x] for x in dat1.keys()]
     dat11=list([dat1[index] for index in v])
     dat11.insert(0,dat1[16]) #patno
     dat11.insert(1,current-dat1[0]) #age
     dat11.insert(1,dat1[7]) #gender
     dat11=[float(x) for x in dat11]
     dat.append(dat11)

for i in range(len(config)):
     dat1=config[i]
     dat1=[dat1[x] for x in dat1.keys()]
     dat11=list([dat1[index] for index in bs])
     dat11.insert(0,dat1[16]) #patno
     dat11.insert(1,current-dat1[0]) #age
     dat11.insert(1,dat1[7]) #gender
     dat11=[float(x) for x in dat11]
     datbl.append(dat11)
     
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


          
dat=np.array(dat)
datbl=np.array(datbl)
dat_con=np.array(dat_con)

mean=np.mean(dat_con, axis=0)
std=np.std(dat_con, axis=0)

Seadl_std=np.std(datbl, axis=0)
std[9]=Seadl_std[9] #SEADL from baseline values


A=np.zeros((n,m))#random.randint(100, size=(n, n))
for i in range(1,n+1): #runs over all symptoms except pat no.
          for k in range(m): #runs over all people
               if i==1:
                  A[i-1][k]=abs(dat[k][i]) #connected only to male
               z=(dat[k][i]-mean[i])/std[i] 
               if abs(z)>=th:
                  if i ==1: #gender
                       continue
                  elif (z*dir[i]>=th):
                       A[i-1][k]+=abs(z)
               
                    
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
print(val)


edges = G3.edges()
weights = [G3[u][v]['weight'] for u,v in edges]
cmap=cm.jet

X3, Y3 = nx.bipartite.sets(G3)
x.append(np.unique(val[:len(X3)]))
noofcomm.append(len(x[2]))
total+=len(x[2])
for i in range(len(x[2])):
            place=np.where(val[:len(X3)]==x[2][i])
            print (place)
            commnodes.append(list(subnodelist[j] for j in place[0]))
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
m=len(config)



for i in range(len(config)):
     dat1=config[i]
     dat1=[dat1[x] for x in dat1.keys()]
     dat11=list([dat1[index] for index in v])
     dat11.insert(0,dat1[16]) #patno
     dat11.insert(1,current-dat1[0]) #age
     dat11.insert(1,dat1[7]) #gender
     dat11=[float(x) for x in dat11]
     dat.append(dat11)

for i in range(len(config)):
     dat1=config[i]
     dat1=[dat1[x] for x in dat1.keys()]
     dat11=list([dat1[index] for index in bs])
     dat11.insert(0,dat1[16]) #patno
     dat11.insert(1,current-dat1[0]) #age
     dat11.insert(1,dat1[7]) #gender
     dat11=[float(x) for x in dat11]
     datbl.append(dat11)
     
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


          
dat=np.array(dat)
datbl=np.array(datbl)
dat_con=np.array(dat_con)

mean=np.mean(dat_con, axis=0)
std=np.std(dat_con, axis=0)

Seadl_std=np.std(datbl, axis=0)
std[9]=Seadl_std[9] #SEADL from baseline values



A=np.zeros((n,m))#random.randint(100, size=(n, n))
for i in range(1,n+1): #runs over all symptoms except pat no.
          for k in range(m): #runs over all people
               if i==1:
                  A[i-1][k]=abs(dat[k][i]) #connected only to male
               z=(dat[k][i]-mean[i])/std[i] 
               if abs(z)>=th:
                  if i ==1: #gender
                       continue
                  elif (z*dir[i]>=th):
                       A[i-1][k]+=abs(z)
                  
random.seed(42)

year4=A
pat4=dat[:,0]

B=np.zeros((n,n))
C=np.zeros((m,m))
D=np.hstack((B,A))
E=np.hstack((A.T,C))
F=np.vstack((D,E))

G4= nx.Graph(F)

part = community.best_partition(G4)
val = [part.get(node) for node in G4.nodes()]
mod = community.modularity(part,G4)
vall.append(val)
print(val)

edges = G4.edges()
weights = [G4[u][v]['weight'] for u,v in edges]
cmap=cm.jet

X4, Y4 = nx.bipartite.sets(G4) # symptomgroups patientgroups
x.append(np.unique(val[:len(X4)]))
noofcomm.append(len(x[3]))
total+=len(x[3])
for i in range(len(x[3])):
            place=np.where(val[:len(X4)]==x[3][i])
            commnodes.append(list(subnodelist[j] for j in place[0]))
            #commcolor.append(G2.vs[place[0]][color])




n=len(dir)

commsize=np.zeros(total)

W=np.zeros((total,total))

track=50
pat_track = pat1[:track]#int(input('Enter a the patient number of the patent you wish to track: '))
pat_traj={}
   
pat_in_comm={}
source={}
target={}
for i in Y1: #year 1 to year 2 connection
      k=vall[0][i] #which comm is pat 1 in
      coms=np.where(x[0]==k)[0]
      patid=pat1[i-18]
      if k not in pat_in_comm:
         pat_in_comm[k]=[]
      pat_in_comm[k].append(patid)
      commsize[k]+=1
      yo=np.where(pat2==patid)[0] #where is the patient in year 2
      if not yo:
         continue
      else:
         k2=vall[1][yo[0]+18]
         comd=np.where(x[1]==k2)[0]      
         W[coms,comd+noofcomm[0]]+=1
         W[comd+noofcomm[0],coms]+=1
         for h in range(len(pat_track)):
            if int(patid)==pat_track[h]:
               source[patid]=[coms,-1,-1]
               target[patid]=[comd+noofcomm[0],-1,-1]
         pat_traj[patid]=[coms,-1,-1,-1]

            

for i in Y2: #year 2 to year 3
      k=vall[1][i]
      coms=np.where(x[1]==k)[0]
      patid=pat2[i-18]
      if k+noofcomm[0] not in pat_in_comm:
         pat_in_comm[k+noofcomm[0]]=[]
      pat_in_comm[k+noofcomm[0]].append(patid)
      commsize[noofcomm[0]+k]+=1
      yo=np.where(pat3==patid)[0] #where is the patient in year 3
      if not yo:
         continue
      else:
         k2=vall[2][yo[0]+18]
         comd=np.where(x[2]==k2)[0]      
         W[coms+noofcomm[0],comd+noofcomm[0]+noofcomm[1]]+=1
         W[comd+noofcomm[0]+noofcomm[1],coms+noofcomm[0]]+=1
         for h in range(len(pat_track)):
            if patid==pat_track[h]:
               if patid in source:
                  source[patid][1]=coms+noofcomm[0]
                  target[patid][1]=comd+noofcomm[0]+noofcomm[1]
         if patid in pat_traj:
            pat_traj[patid][1]=coms+noofcomm[0]
         else:
            pat_traj[patid]=[-1,coms+noofcomm[0],0,0]

            
for i in Y3: #year 3  to year 4
      k=vall[2][i]
      coms=np.where(x[2]==k)[0]
      patid=pat3[i-18]
      if k+noofcomm[0]+noofcomm[1] not in pat_in_comm:
         pat_in_comm[k+noofcomm[0]+noofcomm[1]]=[]
      pat_in_comm[k+noofcomm[0]+noofcomm[1]].append(patid)
      commsize[noofcomm[0]+noofcomm[1]+k]+=1
      yo=np.where(pat4==patid)[0] #where is the patient in year 4
      if not yo:
         continue
      else:
         k2=vall[3][yo[0]+18]
         comd=np.where(x[3]==k2)[0]      
         W[comd+noofcomm[0]+noofcomm[1]+noofcomm[2],coms+noofcomm[0]+noofcomm[1]]+=1
         W[coms+noofcomm[0]+noofcomm[1],comd+noofcomm[0]+noofcomm[1]+noofcomm[2]]+=1
         #print (coms+noofcomm[0]+noofcomm[1], comd+noofcomm[0]+noofcomm[1]+noofcomm[2])
         
         for h in range(len(pat_track)):
            if patid==pat_track[h]:
               if patid in source:
                  source[patid][2]=coms+noofcomm[0]+noofcomm[1]
                  target[patid][2]=comd+noofcomm[0]+noofcomm[1]+noofcomm[2]
         if patid in pat_traj:
            pat_traj[patid][2]=coms+noofcomm[0]+noofcomm[1]
         else:
            pat_traj[patid]=[-1,-1,coms+noofcomm[0]+noofcomm[1],0]



for i in Y4:
   k=vall[3][i]
   coms=np.where(x[3]==k)[0]
   patid=pat4[i-18]
   if k+noofcomm[0]+noofcomm[1]+noofcomm[2] not in pat_in_comm:
         pat_in_comm[k+noofcomm[0]+noofcomm[1]+noofcomm[2]]=[]
   pat_in_comm[k+noofcomm[0]+noofcomm[1]+noofcomm[2]].append(patid)
   commsize[noofcomm[0]+noofcomm[1]+noofcomm[2]+k]+=1
   if patid in pat_traj:
      pat_traj[patid][3]=coms+noofcomm[0]+noofcomm[1]+noofcomm[2]
   #else:
      #pat_traj[patid]=[-1,-1,-1,coms+noofcomm[0]+noofcomm[1]+noofcomm[2]]

q=[]
for k,v in pat_traj.items():
     for thing in v:
        if type(thing)==int:
           if k not in q:
              q.append(k)
        elif len(thing)==0:
           if k not in q:
              q.append(k)
for i in range(len(q)):
   del pat_traj[q[i]] #delete all incomplete trajectories.+noofcomm[1]
q=[]
for k,v in source.items():
     for thing in v:
        if type(thing)==int:
           if k not in q:
              q.append(k)
        elif len(thing)==0 :
           if k not in q:
              q.append(k)
for i in range(len(q)):
   del source[q[i]]
   del target[q[i]]
   

comm_profile=[]#np.zeros((total, len(subnodelist)-1))
std_profile=[]
std=np.array(std[1:])
for i in range(total):
   if i < noofcomm[0]:
      b=[s for s, val in enumerate(pat1) if val in set(pat_in_comm[i])]
      x=year1[:,b]
      comm_profile.append(np.mean(year1[:,b]/commsize[i], axis=1)) #normalize by total number of people in the community
      std_profile.append(np.std(year1[:,b], axis=1))
   elif i < noofcomm[0]+ noofcomm[1]:
      b=[s for s, val in enumerate(pat2) if val in set(pat_in_comm[i])]
      comm_profile.append(np.mean(year2[:,b], axis=1))
      std_profile.append(np.std(year2[:,b], axis=1))
   elif i < noofcomm[0]+ noofcomm[1]+noofcomm[2]:
      b=[s for s, val in enumerate(pat3) if val in set(pat_in_comm[i])]
      comm_profile.append(np.mean(year3[:,b], axis=1))
      std_profile.append(np.std(year3[:,b], axis=1))
   elif i < noofcomm[0]+ noofcomm[1]+noofcomm[2]+noofcomm[3]:
      b=[s for s, val in enumerate(pat4) if val in set(pat_in_comm[i])]
      comm_profile.append(np.mean(year4[:,b], axis=1))
      std_profile.append(np.std(year4[:,b], axis=1))


print (commsize)
#Plotting the community profiles

fig, axs = subplots(6,4, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 2, wspace=.5)

axs = axs.ravel()
tot=0
for i in range(4):
   for j in range(noofcomm[i]):
    axs[4*j+i].bar(np.arange(len(comm_profile[tot])), comm_profile[tot], yerr=std_profile[tot])
    axs[4*j+i].set_title(str(commnodes[tot]), fontsize=8)
    if tot==5 or tot==9 or tot==14 or tot==19:
       axs[4*j+i].set_xticks(np.arange(len(subnodelist)))
       axs[4*j+i].set_xticklabels(subnodelist, rotation=90,fontsize=7)
    else:
       axs[4*j+i].set_xticks([])
    tot+=1
for i in [17,21,22,23]:
   fig.delaxes(axs.flatten()[i])

#show()
savefig('variable_profile_in_comm.png')


      

pat_num=list(pat_traj.keys())
pat_val=list(pat_traj.values())



node_sim=np.zeros((total, total))
for i in range(total):
   for j in range(total):
      count=0
      s=list(pat_traj.values())
      x=[]
      y=[]
      for l in range(len(s)):
         s[l]=[v[0] for v in s[l]]
         if i in s[l]:
            x.append(l)
         if j in s[l]:
            y.append(l)
      #x=np.where(any(e==i for e in s)) #np.where(s==i)
      if not x or not y:
         continue
      #x=[s[l] for l in x]
      x=[pat_traj[list(pat_traj.keys())[b]] for b in x]# all trajectories going through this node
      y=[pat_traj[list(pat_traj.keys())[b]] for b in y]

      '''#method1 no normalization
      for m in range(len(x)): # iterating through all trajectories g
         for n in range(len(y)):
            for k in range(3):
               if x[m][k][0]==y[n][k][0] and x[m][k+1][0]==y[n][k+1][0]:
                  count+=1 #normalizing m*n total possible trajectories times 3 total edges in one trajectory
      node_sim[i,j]=node_sim[j,i]=count/(3*len(x)*len(y))
      v=np.diag(node_sim)'''
      
      
      #method 2 using intersection set over union set with repetitions /multi-set
      yo1=[]
      yo2=[]
      f=0
      for m in range(len(x)):
         for k in range(3):# iterating through all trajectories g
            yo1.append((x[m][k][0],x[m][k+1][0]))

         
      for n in range(len(y)):
         for k in range(3):
            yo2.append((y[n][k][0],y[n][k+1][0]))


      #tot=len(yo1)+len(yo2)
      tot=len(list(set(yo1).union(set(yo2))))
      com=len(list(set(yo1).intersection(set(yo2))))
      '''rep = []
      for elem in deepcopy(yo1):
         if elem in yo2:
            yo1.pop(yo1.index(elem))
            rep.append(yo2.pop(yo2.index(elem)))'''
      if tot!=0:
         node_sim[i,j]=node_sim[j,i]=com/tot


print(np.diag(node_sim), 'k')
      
traj=np.zeros((len(pat_num),len(pat_num)))
for i in range(len(pat_num)):
   for j in range(len(pat_num)):
      for k in range(4):
         traj[i][j]+=node_sim[pat_val[i][k],pat_val[j][k]]

traj=traj/4 #normalizing from all 4 nodes


H=nx.Graph(traj)              
part = community.best_partition(H)
values = [part.get(node) for node in H.nodes()]
mod = community.modularity(part,H)

nocomm=len(np.unique(values))
pat_comm = dict(zip(pat_num, values))

with open('bipartite_pat_dict.pickle', 'wb') as handle:
    pickle.dump(pat_comm, handle, protocol=pickle.HIGHEST_PROTOCOL)


print ('Patients classified into communities based on trajectory: ' , pat_comm)
print (values)

#sort_pat = collections.OrderedDict(sorted(pat_comm.items())) #order pat_num acc to comm. Patients in same community are next to each other in array




figure()    
G=igraph.Graph.Weighted_Adjacency(W.tolist(),mode="undirected")
layout=[]
for i in range (4):
   noc=noofcomm[i]
   for j in range(noc):
      layout.append((i*3,j*4+i/2))
      #pos.append[(i*3,j*2)]


G.vs['label']=commnodes

G.es['width']=[w/10 for w in G.es['weight']]
pal = igraph.RainbowPalette(n=total)
G.vs['color']=[pal.get(i) for i in range(total)]
pal1 = igraph.GradientPalette("red", "white", track)
yo1=[pal1.get(i) for i in range(track)]
pal2 = igraph.GradientPalette("blue", "white", track)
yo2=[pal2.get(i) for i in range(track)]
pal3 = igraph.GradientPalette("green", "white", track)
yo3=[pal3.get(i) for i in range(track)]
pal4 = igraph.GradientPalette("yellow", "orange", track)
yo4=[pal4.get(i) for i in range(track)]

for i in range(len(source)):

   for k in range(3):
      patnid=list(source.keys())[i]

      if pat_comm[list(source.keys())[i]]==0:

         G.add_edge(source[patnid][k][0],target[patnid][k][0],weight = 20, color=yo1[i])
      if pat_comm[list(source.keys())[i]]==1:
         G.add_edge(source[patnid][k][0],target[patnid][k][0],weight = 20, color=yo2[i])
      if pat_comm[list(source.keys())[i]]==2:
         G.add_edge(source[patnid][k][0],target[patnid][k][0],weight = 20, color=yo3[i])
      if pat_comm[list(source.keys())[i]]==3:
         G.add_edge(source[patnid][k][0],target[patnid][k][0],weight = 20, color=yo4[i])



igraph.plot(G,'trajectory_sim_higheroder.png',labels=True, layout=layout,vertex_size=commsize, vertex_color=G.vs['color'], mark_groups=True, bbox=(1324,1524), margin=250)
'''sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=min(weights), vmax=max(weights)))
sm._A = []
clb=colorbar(sm, shrink =0.7)
clb.ax.set_title('Edge weights', fontsize=14)
#show()
'''






'''
commbelong=[]
commdeg=[]
x=np.unique(membership)
whichcomm=[]
total=0 #total number of communities over all years
W=[]


#15August
noofcomm=[]
commsize=[]
commnodes=[]
commcolor=[]
for k in range(4): #year timeline
   W.append([])
   x=np.unique(membership[nosym*k:(k+1)*nosym])
   noofcomm.append(len(x))
   total+=len(x)
   if k==0:
      ppl2=year1
   if k==1:
      ppl2=year2
   if k==2:
      ppl2=year3
   if k==3:
      ppl2=year4
   ppl=ppl2
   print(year1[4])
   ppl[:18]=ppl2[nodeshuff[nosym*k:(k+1)*nosym]] #in order of recent multilayer plotting.
   thresh=0.25
   ppl[abs(ppl)>=thresh]=1
   ppl[abs(ppl)<thresh]=0
   ppl=np.array(ppl)
   ppl = ppl[:, (ppl != 0).any(axis=0)]
   #print(year1[4],'yo',ppl[4],k,'ppl')
   pat=len(ppl[0])
   W[k]=np.zeros((len(x),pat))
   for i in range(len(x)): #community
     place=np.where(membership[nosym*k:(k+1)*nosym]==x[i])[0]
     
     print (place,'place')
     commsize.append(len(place))
     #print(sub[nosym*k:(k+1)*nosym][place],'sa')
     commnodes.append(list(sub[nosym*k:(k+1)*nosym][i] for i in place))
     commcolor.append(G.vs[place[0]+(k*nosym)]['color'])
     for j in range(pat): # number of patients in that year
         for m in range(len(place)):
             W[k][i][j]+=ppl[place[m]][j]
         
         W[k][i][j]=W[k][i][j]/len(place)
         if isnan(W[k][i][j]):
            print('ABORT')
         s=W
   s[k]=W[k]/np.sum(W[k], axis=0)   #making each individuals contribution to all communities in a year ==1


s=np.array(s)
      
nodesize=[list(s[k].sum(axis=1)) for k in range(4)] #sum over individuals gives total individuals (node size) of each community

nodesize=[j for i in nodesize for j in i]
print(len(nodesize), total, nodesize)
conn=np.zeros((total,total))
l=0
for i in range(total):
    for j in range(total):
        conn[i][j]=nodesize[i]*nodesize[j]
pos=[]
for k in range(4):
   i=noofcomm[k]
   for j in range(i):
      pos.append([4*k,2.6*k+2*j])
      
tots=[]      
for k in range(4):
    nodes=noofcomm[k]
    conn[l:l+nodes,l:l+nodes]=0 #no links between communities in the same year
    l+=nodes
    tots.append(l)

conn[:tots[0],tots[1]:]=0 #each community can only connect to the next year
conn[tots[0]:tots[1],tots[2]:]=conn[tots[0]:tots[1],:tots[0]]=0
conn[tots[1]:tots[2],:tots[1]]=0
conn[tots[2]:,:]=0


for row in conn:
   row[row<max(row)]=0 #only strongest connection from each comm to next layer


print(noofcomm,'fd', commcolor, 'fd',commsize, 'fd',commnodes,'ds')
    
H=igraph.Graph.Weighted_Adjacency(conn.tolist(),mode="undirected")
edge=[w/10000 for w in H.es["weight"]]

avg=np.mean(edge)
edge=np.array(edge)
#edge[edge<avg]=0

H.vs['label']=commnodes #find the node label sequence
#pal = igraph.drawing.colors.ClusterColoringPalette(len(membership))
H.es["width"] = edge#[w/20000 for w in H.es["weight"]]
H.vs['color'] = commcolor
print (membership, 'li', len(membership))
#shapee=['rectangle','rectangle','circle','square','triangle','circle','circle','circle','triangle-down','square','polygon','circle','circle','triangle','diamond','diamond','diamond','diamond']
shape=[shapee[c] for c in nodeshuff]
H.vs['size']=nodesize
igraph.plot(H, 'diseasetrajectory_thresh'+str(thresh)+'stddev.png',layout=pos,labels=True, mark_groups=True,margin=170, bbox=(800,1000))
'''










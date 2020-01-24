import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import scipy
import random
import pickle
from matplotlib export *
from numpy import *
import math
import networkx as nx
from networkx.generators.classic import empty_graph, path_graph, complete_graph
import json
import pandas as pd
import igraph as ig
import community
matplotlib.rc('xtick', labelsize=13) 
matplotlib.rc('ytick', labelsize=13)



#############find patients with data across all years ###############
p1=[]
p2=[]
p3=[]
p4=[]
ver=4
config = json.loads(open('bl_v0'+str(ver)+'.json').read())
for i in range(len(config)):
   p1.append(config[i][[x for x in config[i].keys()][16]]) #patno
ver=6
config = json.loads(open('bl_v0'+str(ver)+'.json').read())
for i in range(len(config)):
   p2.append(config[i][[x for x in config[i].keys()][16]]) #patno
ver=8
config = json.loads(open('bl_v0'+str(ver)+'.json').read())
for i in range(len(config)):
   p3.append(config[i][[x for x in config[i].keys()][16]]) #patno
ver=10
config = json.loads(open('bl_v0'+str(ver)+'.json').read())
for i in range(len(config)):
   p4.append(config[i][[x for x in config[i].keys()][16]]) #patno

p=[p1,p2,p3,p4]
yrs=len(p)+1
pat_set=set(p[0]).intersection(*p)
pat_num=list(pat_set)


#uncomment to load dna data
#dna = json.loads(open('dna_all_patients.json').read())

#list of variables
subnodelist = ['gender', 'age','JOLO','SDM','SFT','HVLT','LNS','MOCA','SEADL','RBDQ','ESS','SCOPA-AUT','GDS','STAI','UPDRS1','UPDRS2','UPDRS3','T-UPDRS']
subnodelist_split=deepcopy(subnodelist)

dir=[0,1,1,-1,-1,1,1,1,-1,-1,1,1,1,1,1,1,1,1,1] #direction of disease progression for the variables: 0 is for patient number
   
####### convert raw data json files into matrix form for all years ####
years=4 # data available over number of years. json file for each also has baseline values repeated across json file for all years.
ver_c=[4,6,8,10]

year_mat=[]
pat_mat=[]
for y in range(4): #across all year json files
   ver=ver_c[y]
   config = json.loads(open('bl_v0'+str(ver)+'.json').read())
   dat=[]
   datbl=[]
   current =2018


   v=[2,22,24,9,11,13,15,18,4,20,6,26,28,30,32,34] #organized by domains
   bs=[1,21,23,8,10,12,14,17,3,19,5,25,27,29,31,33]

   #load variable data in year i
   for i in range(len(config)):
        dat1=config[i]
        dat1=[dat1[x] for x in dat1.keys()]
        if dat1[16] in pat_set:
           dat11=list([dat1[index] for index in v]) #variable bs/v determine if matrix formed is baseline data or year y data.
           dat11.insert(0,dat1[16]) #patno
           dat11.insert(1,current-dat1[0]) #age
           dat11.insert(1,dat1[7]) #gender   for gender value 2=male, 1=female
           dat11=[float(x) for x in dat11]
           dat.append(dat11)

   #load baseline data stored in year y json file
   for i in range(len(config)):
        dat1=config[i]
        dat1=[dat1[x] for x in dat1.keys()]
        dat11=list([dat1[index] for index in bs])
        if dat1[16] in pat_set:
           dat11.insert(0,dat1[16])        #patno
           dat11.insert(1,current-dat1[0]) #age
           dat11.insert(1,dat1[7])         #gender
           dat11=[float(x) for x in dat11]
           datbl.append(dat11)



   dat=np.array(dat)
   datbl=np.array(datbl)

   if y==0:
      dat=datbl                            #load baseline data in first loop


   n=shape(dat)[1]-1                       #number of variables excluding pat numbe
   m=shape(dat)[0]                         #number of years

   mean=np.mean(datbl, axis=0)
   std=np.std(datbl, axis=0)


   th_p=50                                 
   th_=np.percentile(datbl, th_p, axis=0)  #setting the threshold to be the median of the baseline ppltn
   th=(th_-mean)/std


   A=np.zeros((n,m))
   for i in range(1,n+1):                   #runs over all symptoms ignoring patient number
          for k in range(m):                #runs over all people
               if i==1 and dat[k][i] ==2:   #connected only to male 
                  A[i-1][k]=1               #controls edge weight
                  continue
               if dir[i]==1:                #variables with higher value = more disease progression
                  if (z>=th[i]):
                     A[i-1][k]+=1
               if dir[i]==-1:               #variables with higher value = less disease progression
                  if (z<=th[i]):
                     A[i-1][k]+=1

   year_mat.append(A)                       #store edge weights
   pat_mat.append(dat[:,0])                 #store patient numbers




#remove patients with incomplete data

yrs=len(pat_mat)
pat_set=set(pat_mat[0]).intersection(*pat_mat)   
pat_num=np.sort(list(pat_set))

pat_all=[]
for yy in range(yrs): 
   pat_all.append([i for i, val in enumerate(pat_mat[yy]) if val in pat_set])


#create list F containing binarized matrices for each year with patient-variable data. F is an ndarray with shape yearsxvariablesxpatients
F=[]
for yy in range(yrs): 
   F.append(year_mat[yy][:,pat_all[yy]])

F_=np.array(F)                                      

                                                                    
c=int(shape(F_)[2]*0.8)                             #80% of patients are training set
F_test=F_[:,:,c:]                                   #test-patient data
F_=F_[:,:,:c]                                       #train-patient data
test_pat_num=pat_num[c:]
pat_num=pat_num[:c]

num_p=len(pat_num)
M=np.zeros((num_p,num_p))


#first order patient-patient adjacency matrix
for i in range(num_p):
   for j in range(num_p):
      M[i][j]=(F_[:,:,i]==F_[:,:,j]).sum()           #number of common elements in the trajectory profile of patient i and j


#community (subtype) detection on patient-patient graph                                             
H=nx.Graph(M)              
part = community.best_partition(H)
values = [part.get(node) for node in H.nodes()]
mod = community.modularity(part,H)

nocomm=len(np.unique(values))                 
pat_comm = dict(zip(pat_num, values))               #pat_number and which community/subtype they belong in


# group patient profiles for patients in each community/subtype:
uni=np.unique(values)
no_of_ppl=[]

prof=[]
for i in range(len(uni)):
   prof.append([])
   zorro=np.where(values==uni[i])[0]
   for s in range(len(zorro)):
      prof[i].append(F_[:,:,zorro[s]])              #genes are binarized
   no_of_ppl.append(shape(prof[i])[0])
   prof[i]=np.mean(prof[i],axis=0)                  #mean profile of each community 
   

print (no_of_ppl, ':number of ppl in each comm')
print (len(no_of_ppl), ':number of communities')

a=np.mean(F_,axis=2)

 

#delete communities with less than a threshold number of patients
all_prof=np.array(deepcopy(prof)) #copy all communities
all_ppl=np.array(deepcopy(no_of_ppl))

rem=[]
for i in range(len(no_of_ppl)):
   if no_of_ppl[i]<=10:
      rem.append(i)
prof=np.delete(prof, rem, axis=0)

comm=shape(prof)[0]                                  #number of significant communities
no_of_ppl=np.delete(no_of_ppl, rem, axis=0)



###swapping positions of some variables so variables in the same domain are plotted next to each other 

def swap_cols(arr, frm, to):
    arr[:,[frm, to]] = arr[:,[to, frm]]

for i in range(len(prof)):
   prof[i,:,[4,7]] = prof[i,:,[7,4]] #swap position of SFT and MOCA in plot
   prof[i]=np.roll(prof[i],4,axis=1)
   prof[i,:,[4,0]] = prof[i,:,[0,4]]
   prof[i,:,[1,5]] = prof[i,:,[5,1]]
   prof[i,:,[4,2]] = prof[i,:,[2,4]]
   prof[i,:,[5,3]] = prof[i,:,[3,5]]
a[:,[4,7]] = a[:,[7,4]] #swap position of SFT and MOCA in plot
a=np.roll(a,4,axis=1)
a[:,[4,0]] = a[:,[0,4]]
a[:,[1,5]] = a[:,[5,1]]
a[:,[4,2]] = a[:,[2,4]]
a[:,[5,3]] = a[:,[3,5]]

for i in range(len(all_prof)):
   all_prof[i,:,[4,7]] = all_prof[i,:,[7,4]] #swap position of SFT and MOCA in plot
   all_prof[i]=np.roll(all_prof[i],4,axis=1)
   all_prof[i,:,[4,0]] = all_prof[i,:,[0,4]]
   all_prof[i,:,[1,5]] = all_prof[i,:,[5,1]]
   all_prof[i,:,[4,2]] = all_prof[i,:,[2,4]]
   all_prof[i,:,[5,3]] = all_prof[i,:,[3,5]]

names = ['gender', 'age','MDS-UPDRS1','MDS-UPDRS2','MDS-UPDRS3','T-MDS-UPDRS','JOLO','SDM','MoCA','HVLT','LNS','SFT','SEADL','RBDQ','ESS','SCOPA-AUT','GDS','STAI']


######plotting the community profiles #############


figure()
fig, axes = subplots(nrows=comm+1, ncols=1) #if you need to plot cbar, plot here and then crop out common colorbar
i=0
for ax in axes.flat:
    ax.set_yticks(range(0,yrs))
    if i!=comm:
       im = ax.imshow(prof[i],cmap='Greys',vmin=0,vmax=1)
       ax.set_ylabel('Yrs \n'+str(no_of_ppl[i])+ ' ppl',fontsize=12)
       ax.set_xticks([])
       i+=1
    else:
       im = ax.imshow(a,cmap='Greys',vmin=0,vmax=1)
       ax.set_ylabel('Total pptn',fontsize=12)
       ax.set_xticks(np.arange(len(names)))
       ax.set_xticklabels(names, rotation=90,fontsize=12)

xlabel('Variables',fontsize=16)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
savefig('average_community_variable_profile.pdf', bbox_inches='tight', minor=True)

from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import math
colWithCategories=[0,1,2,3,4,18,19,20,23,24,25,28,31,34,35,41,43,47,51,52,54,55,56,59,60,61,64,66,67,70]
dataset=pd.read_csv('train1.csv')
X=dataset.iloc[:,:-1].values;
y=dataset.iloc[:,82].values;
X_check=pd.DataFrame(X)
dataPoints=X.shape[0]
features=X.shape[1];

#Preprocessing
ordinalEncoder=preprocessing.OrdinalEncoder()
ordinalList=[ordinalEncoder for i in range(dataPoints)]
for feature in colWithCategories:
    o=preprocessing.OrdinalEncoder()
    missingValueImputer=SimpleImputer(missing_values=np.nan,strategy="constant", fill_value=0)
    X[:,feature]=missingValueImputer.fit_transform(X[:,feature].reshape(-1,1)).reshape(-1,)
    X[:,feature]=o.fit_transform(X[:,feature].reshape(-1,1)).reshape(-1,)
    ordinalList[feature]=o;

missingValueImputer=SimpleImputer(missing_values=np.nan,strategy="constant", fill_value=0)
X=missingValueImputer.fit_transform(X)

##apply WOA-kmeans clustering
numberOfCluster=2
numberOfWhale=10
iterations=100

#intialise
import random
from sklearn.metrics.pairwise import euclidean_distances
centresOfwhale=np.zeros((numberOfWhale,numberOfCluster,features))

import time

for whale in range(numberOfWhale):
    for cluster in range(numberOfCluster):
        for feature in range(features):
            centresOfwhale[whale,cluster,feature]=float(random.randint(np.min(X[:,feature]),np.max(X[:,feature])))
bestWhale=0
for iteration in range(iterations):
    print(iteration)
    #dataPointsInCluster=[[[] for cluster in range(numberOfCluster)] for whale in range(numberOfWhale)]
    dataPointsInCluster=np.zeros((numberOfWhale,numberOfCluster))
    bestWhale=0
    bestDist=float("inf")
    startTime=time.time()
    for whale in range(numberOfWhale):
        dist=0.00
        clus1=np.zeros((features))
        clus2=np.zeros((features))
        for dataPoint in range(dataPoints):
            bestEuclidianDist=float("inf")
            bestCluster=0
            for cluster in range(numberOfCluster):
                euclidDist=np.linalg.norm(centresOfwhale[whale,cluster]-X[dataPoint,:])
                if(euclidDist<bestEuclidianDist):
                    bestEuclidianDist=euclidDist
                    bestCluster=cluster
            dist=dist+bestEuclidianDist
            #dataPointsInCluster[whale][bestCluster].append(dataPoint)
            dataPointsInCluster[whale][bestCluster]=dataPointsInCluster[whale][bestCluster]+1
            if bestCluster==0:
                clus1=clus1+X[dataPoint]
            if bestCluster==1:
                clus2=clus2+X[dataPoint]
        if(dist<bestDist):
            bestDist=dist
            bestWhale=whale
        
        if(dataPointsInCluster[whale][0]!=0):
            centresOfwhale[whale][0]=clus1/dataPointsInCluster[whale][0]
        
        if(dataPointsInCluster[whale][1]!=0):
            centresOfwhale[whale][1]=clus1/dataPointsInCluster[whale][1]
    #shift the centroid in the centre of the datapoints inside cluster
    print(time.time()-startTime)
    startTime=time.time();
    '''print("shift started")
    for whale in range(numberOfWhale):
        for cluster in range(numberOfCluster):
            numberOfPoint=len(dataPointsInCluster[whale][cluster])
            if numberOfPoint==0 :
                continue
            for feature in range(features):
                sum=0.00
                for dataPoint in dataPointsInCluster[whale][cluster]:
                    sum=sum+X[dataPoint][feature];
                sum=sum/numberOfPoint
                centresOfwhale[whale,cluster,feature]=sum'''
    
    #shift centroids using equations of WOA
    print(time.time()-startTime)
    startTime=time.time();
    print("woa started")
    a=2-iteration*((2.00)/iterations) #eqn 2.3
    a2=-1+iteration*((-1.00)/iterations)
    for whale in range(numberOfWhale):
        r1=random.random()
        r2=random.random()
        A=2*a*r1-a;  # Eq. (2.3) in the paper
        C=2*r2;      # Eq. (2.4) in the paper
        p=random.random()
        b=1;               #  parameters in Eq. (2.5)
        l=(a2-1)*random.random()+1;   #  parameters in Eq. (2.5)
        
        for cluster in range(numberOfCluster):
            if p<0.5 :
                if abs(A)>=1 :
                    rand_leader_index = int(math.floor((numberOfWhale-1)*random.random()+1));
                    X_rand = centresOfwhale[rand_leader_index]
                    D_X_rand=abs(C*X_rand[cluster]-centresOfwhale[whale,cluster]); # Eq. (2.7)
                    centresOfwhale[whale,cluster]=X_rand[cluster]-A*D_X_rand;      # Eq. (2.8)
                elif abs(A)<1 :
                    D_Leader=abs(C*centresOfwhale[bestWhale,cluster]-centresOfwhale[whale,cluster]); # Eq. (2.1)
                    centresOfwhale[whale,cluster]=centresOfwhale[bestWhale,cluster]-A*D_Leader;      # Eq. (2.2)
            elif p>=0.5 :
                distance2Leader=abs(centresOfwhale[bestWhale,cluster]-centresOfwhale[whale,cluster]);      # Eq. (2.5)
                centresOfwhale[whale,cluster]=distance2Leader*math.exp(b*l)*math.cos(l*2*3.14)+centresOfwhale[bestWhale,cluster];
    print(time.time()-startTime)
    startTime=time.time();
                
y_pred=[0 for dataPoint in range(dataPoints)]
for dataPoint in range(dataPoints):
    d1=np.linalg.norm(centresOfwhale[bestWhale,0]-X[dataPoint,:])
    d2=np.linalg.norm(centresOfwhale[bestWhale,1]-X[dataPoint,:])
    if d1<d2 :
        y_pred[dataPoint]=1
    else:
        y_pred[dataPoint]=0

from sklearn.metrics import confusion_matrix
confusion_matrix(y,y_pred)

from sklearn.metrics import accuracy_score
print(accuracy_score(y,y_pred))

y_pred1=[0 for dataPoint in range(dataPoints)]
for dataPoint in range(dataPoints):
    d1=np.linalg.norm(centresOfwhale[bestWhale,0]-X[dataPoint,:])
    d2=np.linalg.norm(centresOfwhale[bestWhale,1]-X[dataPoint,:])
    if d1<d2 :
        y_pred1[dataPoint]=0
    else:
        y_pred1[dataPoint]=1
            
            
from sklearn.metrics import confusion_matrix
confusion_matrix(y,y_pred1)

from sklearn.metrics import accuracy_score
print(accuracy_score(y,y_pred1))
                
                
            
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
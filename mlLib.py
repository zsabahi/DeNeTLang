# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 20:57:58 2019

@author: Zahra Alimadadi
"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
import warnings


##
# Standardizing train dataset, it dump the standard model if it called from the clustering_Kmeans method. It is when it is given an directory 
##
def standardize_fit(x_train, standard):
    
    sc_X = StandardScaler()
    x_train = sc_X.fit_transform(x_train)
    
    if standard !=0:
        dump(sc_X, standard)
    
    return x_train

##
# This method transform test dataset based on train dataset standardization
##
def standardize_transform(x_test, standard):
    
    sc_X = load(standard)
    x_test = sc_X.transform(x_test)
    
    return x_test


##
# Compute wcss for a given range, started from j
# Return wcss and related range each as a list
##
def computewcss(j, x_train, rang):
    wcss = [] 
    x = []
    # to compute with in cluster some of squre
    if rang < 252:
        k = [i for i in range(j,rang)]
    if rang < 502:
        k = [i for i in range(j, rang, 2)]
    elif rang < 1002:
        k = [i for i in range(j, rang, 5)]
    else:
        k = [i for i in range(j, rang, 15)] 
#    if rang > 100:
#        k.extend([i for i in range(101,rang,step)])
#        
#    k = [i for i in range(j, min(rang, 100), 2)]
        
    for i in k:
        if i > 100 and i%50 == 1:
            print(i)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("error")
            try:   
                kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42) #max_iter =300 , m_init =10
                kmeans.fit(x_train) 
                wcss.append(kmeans.inertia_)
                x.append(i)
            except Exception as e:
                message = str(e)
                if "Number of distinct clusters" in message: # ConvergenceWarning                
                    rang = int(message[message.find("(")+1:message.find(")")])
                    print("catched ", rang)
                return wcss, x
    return wcss, x

##
# Determine the range for which the clustering should be done for elbow method
##    
def clusteringRang(j, x_trainlen, rangdefault):
    #active = True
    if rangdefault == 0:
        if x_trainlen<11:
            rang = x_trainlen 
        elif x_trainlen > 1000:
            rang = min (int(max (x_trainlen/10, 30))+1 , 251) # 1501
        elif x_trainlen > 150:
            rang = min (int(max (x_trainlen/20, 30))+1 , 151) # 501             
        else:
            rang = 11  
    else:
        rang = 1001 
        
    return rang   
        
##
# Determine the number of cluster for each proto or all the dataset using KneeLocator.
# The elbow diagram is drawn here.
##    
def elbow(x_train, proto, figdir):   
    # print(proto)
    x_train = standardize_fit(x_train, 0)
    ncluster = 1
    if proto == "all":
        rang = clusteringRang(1, len(x_train), 1)
        wcss, x = computewcss(1, x_train, rang)
    else:
        rang = clusteringRang(1, len(x_train), 0)
        wcss, x = computewcss(1, x_train, rang)
   
    with open(figdir+"/cluster.txt", "a") as rfile:        
        if len(wcss) > 1:   
            min_wcss = min(wcss)
            max_wcss = max(wcss)   
            if max_wcss < 500 or min_wcss == max_wcss:            
                ncluster = 1
                # rfile.write("%s: Cluster %s \n" % (proto, ncluster))
            else:
                rang = max(x)
                s = 1
                ncluster = KneeLocator(x, wcss, S=s, curve='convex', direction='decreasing', online=True).knee 
                if ncluster == None:
                    ncluster = 1                   
                gap = wcss[x.index(ncluster)]
                # rfile.write("%s: Cluster %s, Gap %s, S %s, range %s \n" % (proto, ncluster, gap, s, rang))
                while gap < 1000 and proto != "all" and ncluster > 1:
                    ncluster = x[x.index(ncluster)-1]                    
                    gap = wcss[x.index(ncluster)-1]
                    # rfile.write("\t %s: Cluster %s, Gap %s, S %s, range %s \n" % (proto, ncluster, gap, s, rang))
                    if gap > 1000:
                        ncluster = x[x.index(ncluster)+1]
                        gap = wcss[x.index(ncluster)+1]
                        # rfile.write("\t %s: Cluster %s, Gap %s, S %s, range %s \n" % (proto, ncluster, gap, s, rang))
                        break      
                #print( proto, gap, s, rang, ncluster)
                if gap > 1000:
                    s = 300
                    prencluster = ncluster
                    ncluster = KneeLocator(x, wcss, S=s, curve='convex', direction='decreasing', online=True).knee #, online=True    

                    if ncluster == None:
                        ind = x.index(prencluster)
                        ncluster= x[ind+1]
                        gap = wcss[ind+1]  
                    else:
                        gap = wcss[x.index(ncluster)]
                    # rfile.write("\t %s: Cluster %s, Gap %s, S %s, range %s \n" % (proto, ncluster, gap, s, rang))

    plt.plot(x, wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of Clusters for %s'%(proto))
    plt.ylabel('WCSS')
    plt.savefig(figdir+"/%s.png"%proto)
    plt.close("all")
    #plt.show()
    return ncluster

##
# Input:
# n : number of cluster
# trainds: train dataset  
# modeldir: where kmeans++ model is stored 
# featureVectorstandarddir: where standardized train ds is stored
# Output: data's clusters
##
def clustering_Kmeans(n, trainDS, modeldir, standarddir):
    
    standardizeddata = standardize_fit(trainDS, standarddir)
    kmeans = KMeans(n_clusters = n, init = 'k-means++', random_state = 42) 
    kmeans_cluster = kmeans.fit_predict(standardizeddata)
    
    dump(kmeans, modeldir) 
    
    return kmeans_cluster   

##
# Input:
# testds: test dataset  
# modeldir: where kmeans++ model is stored 
# featureVectorstandarddir: where standardized train ds is stored
# Output: assigned clusters
##
    
def assignCluster_Kmeans(testDS, modeldir, standarddir):
    
    kmeans = load(modeldir) 
    standardizeddata = standardize_transform(testDS, standarddir)
    kmeans_cluster = kmeans.predict(standardizeddata)
    
    return kmeans_cluster
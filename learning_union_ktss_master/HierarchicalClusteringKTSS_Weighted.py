from matplotlib import pyplot as plt
from matplotlib import rc
from scipy.cluster.hierarchy import linkage, to_tree, fcluster, _append_singleton_leaf_node, _link_line_colors, _plot_dendrogram
from scipy._lib.six import string_types
import numpy as np
from learning_union_ktss_master.KTestable_Weighted import *
from learning_union_ktss_master.DendrogramKTSS import dendrogram_ktss
from sys import maxsize as MAX_DIST



def condensed_index(n, i, j):
    """
    Calculate the condensed index of element (i, j) in an n x n condensed
    matrix.
    """
    if i < j:
        return int( n * i - (i * (i + 1) / 2) + (j - i - 1) )
    elif i > j:
        return int( n * j - (j * (j + 1) / 2) + (i - j - 1) )



class LinkageUnionFind:
    """Structure for fast cluster labeling in unsorted dendrogram."""

    def __init__(self, n):
        self.parent = np.arange(2 * n - 1, dtype=np.intc)
        self.next_label = n
        self.size = np.ones(2 * n - 1, dtype=np.intc)

    def merge(self, x, y):
        self.parent[x] = self.next_label
        self.parent[y] = self.next_label
        size = self.size[x] + self.size[y]
        self.size[self.next_label] = size
        self.next_label += 1
        return size

    def find(self, x):
        p = x

        while self.parent[x] != x:
            x = self.parent[x]

        while self.parent[p] != x:
            p, self.parent[p] = self.parent[p], x

        return x


def label(Z, n):
    """Correctly label clusters in unsorted dendrogram."""
    uf = LinkageUnionFind(n)
    for i in range(n - 1):
        x, y = int(Z[i, 0]), int(Z[i, 1])
        x_root, y_root = uf.find(x), uf.find(y)
        if x_root < y_root:
            Z[i, 0], Z[i, 1] = x_root, y_root
        else:
            Z[i, 0], Z[i, 1] = y_root, x_root
        Z[i, 3] = uf.merge(x_root, y_root)
        # print(Z[i, 0],Z[i, 1])



def ktssDistance(a,b,k_window):

    k = k_window

    E1,I1,F1,T1,C1,W1 = calculateEIFTC([a],k)
    E2,I2,F2,T2,C2,W2 = calculateEIFTC([b],k)

    
    return len( I1.keys() - I2.keys() ) + len( F1.keys() - F2.keys() ) + len( T1.keys() - T2.keys() ) + len( I2.keys() - I1.keys() ) + len( F2.keys() - F1.keys() ) + len( T2.keys() - T1.keys() ) 
    
    
    # E = list(set(E1+E2))

    # if I1.keys()==I2.keys() and F1.keys()==F2.keys() and T1.keys()==T2.keys() and C1.keys()==C2.keys():
        # return 0
    # elif I1.keys()<=I2.keys() and F1.keys()<=F2.keys() and T1.keys()<=T2.keys() and C1.keys()<=C2.keys():
        # return 0.5
    # elif I1.keys()>=I2.keys() and F1.keys()>=F2.keys() and T1.keys()>=T2.keys() and C1.keys()>=C2.keys():
        # return 0.5
    # else:

        # card1 = len(I1) + len(F1) + len(T1) + len(C1)
        # card2 = len(I2) + len(F2) + len(T2) + len(C2)

        # I = list(set(list(I1.keys())+list(I2.keys())))
        # F = list(set(list(F1.keys())+list(F2.keys())))
        # T = list(set(list(T1.keys())+list(T2.keys())))
        # C = list(set(list(C1.keys())+list(C2.keys())))

        # card0 = len(I) + len(F) + len(T) + len(C)

        # return abs( card0 - max(card1,card2) )


def ktssDistance2(a,b,k_window):

    k = k_window

    E1 = a.E
    I1 = a.I
    F1 = a.F
    T1 = a.T
    C1 = a.C
    E2 = b.E
    I2 = b.I
    F2 = b.F
    T2 = b.T
    C2 = b.C
    
    
    return len( I1.keys() - I2.keys() ) + len( F1.keys() - F2.keys() ) + len( T1.keys() - T2.keys() ) + len( I2.keys() - I1.keys() ) + len( F2.keys() - F1.keys() ) + len( T2.keys() - T1.keys() ) 

    
    # E = list(set(E1+E2))

    # if I1.keys()==I2.keys() and F1.keys()==F2.keys() and T1.keys()==T2.keys() and C1.keys()==C2.keys():
        # return 0
    # elif I1.keys()<=I2.keys() and F1.keys()<=F2.keys() and T1.keys()<=T2.keys() and C1.keys()<=C2.keys():
        # return 0.5
    # elif I1.keys()>=I2.keys() and F1.keys()>=F2.keys() and T1.keys()>=T2.keys() and C1.keys()>=C2.keys():
        # return 0.5
    # else:

        # card1 = len(I1) + len(F1) + len(T1) + len(C1)
        # card2 = len(I2) + len(F2) + len(T2) + len(C2)

        # I = list(set(list(I1.keys())+list(I2.keys())))
        # F = list(set(list(F1.keys())+list(F2.keys())))
        # T = list(set(list(T1.keys())+list(T2.keys())))
        # C = list(set(list(C1.keys())+list(C2.keys())))

        # card0 = len(I) + len(F) + len(T) + len(C)

        # return abs( card0 - max(card1,card2) )
        
        

def nn_chain_ktss(dists, n, list_ktss_clusters, k_window, crossover_detect):
    notprocessed = {el:None for el in range(n)}
    incase = [el for el in range(n)]
    alive = {el:None for el in range(n)}
    
    Z_arr = np.empty((n - 1, 4))
    Z = Z_arr

    D = dists.copy()  # Distances between clusters.
    size = np.ones(n, dtype=np.intc)  # Sizes of clusters.

    # Variables to store neighbors chain.
    cluster_chain = np.ndarray(n, dtype=np.intc)
    chain_length = 0

    for k in range(n - 1):
        #print('lm',k,'of',n-1)
        
        if chain_length == 0:
            chain_length = 1
            for i in range(n):
                if size[i] > 0:
                    cluster_chain[0] = i
                    break
        # Go through chain of neighbors until two mutual neighbors are found.
        while True:
            x = cluster_chain[chain_length - 1]
            # We want to prefer the previous element in the chain as the
            # minimum, to avoid potentially going in cycles.
            if chain_length > 1:
                y = cluster_chain[chain_length - 2]
                try:
                    current_min = D[condensed_index(n, x, y)]
                except TypeError:
                    #HERE finish the linkage and return the linkage matrix
                    list_nonprocessed = list(notprocessed)
                    index = k
                    for i in range(0,len(list_nonprocessed)-1):
                        try:
                            Z_arr[index, 0] = list_nonprocessed[i]
                            Z_arr[index, 1] = list_nonprocessed[i+1]
                            Z_arr[index, 2] = MAX_DIST
                            Z_arr[index, 3] = size[list_nonprocessed[i]] + size[list_nonprocessed[i+1]]
                        except IndexError:
                            break
                        size[list_nonprocessed[i+1]] = size[list_nonprocessed[i]] + size[list_nonprocessed[i+1]]
                        index += 1
                    order = np.argsort(Z_arr[:, 2], kind='mergesort')
                    Z_arr = Z_arr[order]
                    label(Z_arr, n)
                    return Z_arr

            else:
                current_min = np.Infinity

                
            found = False
            for i in range(n):
            
                if size[i] == 0 or x == i:
                    continue

                dist = D[condensed_index(n, x, i)]
                
                if dist < current_min:
                    if dist<0:
                        continue
                    if dist == 0:
                        found = True
                        current_min = dist
                        y = i
                        break
                    if crossover_detect:
                        #HERE CONDITION ON LANGUAGE MERGABILITY!!!
                        kss_x = list_ktss_clusters[x]
                        kss_i = list_ktss_clusters[i]
                        alph = list(set(kss_x.E+kss_i.E))
                        # print('test',x,i,check_union(alph,kss_x.I,kss_x.F,kss_x.T,kss_i.I,kss_i.F,kss_i.T,k_window))
                        if not check_union(alph,kss_x.I,kss_x.F,kss_x.T,kss_i.I,kss_i.F,kss_i.T,k_window):
                            continue
                    found = True
                    current_min = dist
                    y = i
                    
                
            
            if chain_length > 1 and y == cluster_chain[chain_length - 2]:
                if crossover_detect:
                    kss_x = list_ktss_clusters[x]
                    kss_i = list_ktss_clusters[y]
                    alph = list(set(kss_x.E+kss_i.E))
                    if check_union(alph,kss_x.I,kss_x.F,kss_x.T,kss_i.I,kss_i.F,kss_i.T,k_window):
                        break
                else:
                    break
            
            if not found:
                try:
                    cluster_chain[chain_length] = incase[incase.index(x)+1]
                except IndexError:
                    
                    #HERE finish the linkage and return the linkage matrix
                    list_nonprocessed = list(notprocessed)
                    index = k
                    for i in range(0,len(list_nonprocessed)-1):
                        try:
                            Z_arr[index, 0] = list_nonprocessed[i]
                            Z_arr[index, 1] = list_nonprocessed[i+1]
                            Z_arr[index, 2] = MAX_DIST
                            Z_arr[index, 3] = size[list_nonprocessed[i]] + size[list_nonprocessed[i+1]]
                        except IndexError:
                            break
                        size[list_nonprocessed[i+1]] = size[list_nonprocessed[i]] + size[list_nonprocessed[i+1]]
                        index += 1
                    order = np.argsort(Z_arr[:, 2], kind='mergesort')
                    Z_arr = Z_arr[order]
                    label(Z_arr, n)
                    return Z_arr
                    
                    
                chain_length += 1
                continue
            
            cluster_chain[chain_length] = y
            chain_length += 1
            
            
            
        # Merge clusters x and y and pop them from stack.
        chain_length -= 2

        # This is a convention used in fastcluster.
        if x > y:
            x, y = y, x

        #update the list_ktss_clusters
        kss_x = list_ktss_clusters[x]
        kss_y = list_ktss_clusters[y]
        kss_x_y = fusion_of_kss(kss_x,kss_y)

        list_ktss_clusters[x] = kss_x_y
        list_ktss_clusters[y] = kss_x_y

        # get the original numbers of points in clusters x and y
        nx = size[x]
        ny = size[y]

        # Record the new node.
        Z[k, 0] = x
        Z[k, 1] = y
        try:
            del notprocessed[x]
            incase.remove(x)
        except KeyError:
            pass
        Z[k, 2] = current_min
        Z[k, 3] = nx + ny
        size[x] = 0  # Cluster x will be dropped.
        del alive[x]
        size[y] = nx + ny  # Cluster y will be replaced with the new cluster
        # Update the distance matrix.
        for i in range(n):
            ni = size[i]
            if ni == 0 or i == y:
                continue
            D[condensed_index(n, i, y)] = ktssDistance2(kss_x_y,list_ktss_clusters[i],k_window)

    # Sort Z by cluster distances.
    order = np.argsort(Z_arr[:, 2], kind='mergesort')
    Z_arr = Z_arr[order]
    # Find correct cluster labels inplace.
    label(Z_arr, n)
    return Z_arr




def findThreshold(num_clusters, linkage_matrix):
    threshold = 2.0
    tested_up = MAX_DIST
    tested_down = 0.0
    solution = None
    iteration = 0
    while iteration < 2000:
        iteration += 1
        solution = fcluster(linkage_matrix,threshold,'distance')
        if len(set(solution)) == num_clusters:
            break
        elif len(set(solution)) < num_clusters:
            tested_up = threshold
            threshold = (threshold + tested_down)/2
        else:
            tested_down = threshold
            threshold = (threshold + tested_up)/2
    if threshold == MAX_DIST:
        return MAX_DIST-1
    return threshold




def calculateDistanceMatrix_ktss_distance(dataset,k_window):
    distance_matrix = []
    i = 0
    line = []
    for f1 in dataset:
        #print('dm',i,'of',len(dataset))
        
        j = 0
        for f2 in dataset:
            if i < j:
                line.append(ktssDistance(f1, f2,k_window))
            j += 1
        i += 1
    return line


def generatePlot(linkage_matrix, threshold, listLabels, PATH_PLOT):
    # listLabels = ['$Z_1$','$Z_2$','$Z_3$','$Z_4$','$Z_5$','$Z_6$','$Z_7$','$Z_8$']
    # generate plot
    # rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    # rc('text', usetex=True)
    # plt.figure(figsize=(6, 3))
    plt.figure(figsize=(100, 100))
    truc = dendrogram_ktss(
        linkage_matrix,
        leaf_rotation=90.,  # rotates the x axis labels
        # leaf_rotation=0.,  # rotates the x axis labels
        # leaf_font_size=15.,  # font size for the x axis labels
        leaf_font_size=4.,  # font size for the x axis labels
        color_threshold = threshold,
        labels = listLabels,
        orientation = 'top',above_threshold_color='k'
    )
    #plt.yticks([])
    top = max(linkage_matrix[np.where(linkage_matrix[:,2] <= MAX_DIST/2)][:, 2]) + max(linkage_matrix[np.where(linkage_matrix[:,2] <= MAX_DIST/2)][:, 2]) * 0.01
    plt.ylim(bottom=-0.07,top=top+0.5)
    plt.tight_layout()
    plt.savefig(PATH_PLOT)


def learn_unions_ktss(dataset,num_clusters,k_window,show_plot,crossover_detect,PATH_PLOT):
    # calculate distance matrix
    distance_matrix = calculateDistanceMatrix_ktss_distance(dataset,k_window)
    # from scipy.spatial.distance import squareform
    # print(squareform(distance_matrix))
    #Calculate list of initial clusters
    WArr=[]
    list_ktss_clusters = []
    for cluster in dataset:
        E,I,F,T,C,W = calculateEIFTC([cluster],k_window)
        kss = k_testable(E,I,F,T,C)
        list_ktss_clusters.append(kss)
        WArr.append(W)
        
    if len(list_ktss_clusters) < 2:
        dict_ktss = {}
        WArr=[]

        for cluster in dataset:
            E,I,F,T,C,W = calculateEIFTC([cluster],k_window)
            kss = k_testable(E,I,F,T,C)
            dict_ktss[kss] = None
            WArr.append(W)
        
        return list(""), dict_ktss,WArr
        # return list(""),list_ktss_clusters
    
    # calculate full dendrogram
    linkage_matrix = nn_chain_ktss(distance_matrix, len(dataset),list_ktss_clusters,k_window, crossover_detect)

    #find threshold to split the dendrogram
    threshold = findThreshold(num_clusters, linkage_matrix)
    if threshold > MAX_DIST/2:
        threshold = max(linkage_matrix[np.where(linkage_matrix[:,2] <= MAX_DIST/2)][:, 2]) + max(linkage_matrix[np.where(linkage_matrix[:,2] <= MAX_DIST/2)][:, 2]) * 0.01
    
    #*** Union
#    print("*** threshold is "+str(threshold)+ " ***")
    
    # generatePlot(linkage_matrix,threshold,dataset)
    if show_plot:
        generatePlot(linkage_matrix,threshold,dataset,PATH_PLOT)
        
    clusters = fcluster(linkage_matrix,threshold,'distance')

    # get the strings of each cluster
    dict_clusters = {}
    for clust_nb in range(1,len(set(clusters))+1):
        dict_clusters[clust_nb] = []
        i = 0
        for elt in clusters:
            if elt == clust_nb:
                dict_clusters[clust_nb].append(dataset[i])
            i += 1
            
    dict_ktss = {}
    WArr=[]

    for cluster in dict_clusters:
        E,I,F,T,C,W = calculateEIFTC(dict_clusters[cluster],k_window)
        kss = k_testable(E,I,F,T,C)
        dict_ktss[kss] = None
        WArr.append(W)
        
    return list(clusters), dict_ktss,WArr

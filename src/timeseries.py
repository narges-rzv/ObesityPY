# from: http://nbviewer.jupyter.org/github/alexminnaar/time-series-classification-and-clustering/blob/master/Time%20Series%20Classification%20and%20Clustering.ipynb
import numpy as np
import random
random.seed(1)
import math
try:
    import matplotlib.pyplot as plt
except:
    print('cant plot. install matplotlib if you want to visualize')


def load_temporal_data(xtrain, headers):
    print(headers)
    print(xtrain.shape)
    vital_count = 9
    newh = headers.reshape(vital_count, 11)
    newx = xtrain.reshape(xtrain.shape[0], vital_count, 11)
    return newx[:,1:,:], newh[1:,:]


def euclid_dist(t1,t2):
    return np.sqrt(sum((t1-t2)**2))

def euclid_dist_w_missing(t1, t2):
    nonzeros = (t1 != 0) & (t2 != 0)
    # print(nonzeros.sum())
    if nonzeros.sum() == 0:
        return float('inf')
    # print( (t1[nonzeros]-t2[nonzeros])**2)
    return np.sqrt(sum((t1[nonzeros]-t2[nonzeros])**2))/nonzeros.sum()

def LB_Keogh(s1,s2,r):
    LB_sum=0
    for ind,i in enumerate(s1):
        
        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        
        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2
    
    return np.sqrt(LB_sum)
def DTWDistance(s1, s2,w):
    DTW={}
    
    w = max(w, abs(len(s1)-len(s2)))
    
    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0
  
    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
        
    return np.sqrt(DTW[len(s1)-1, len(s2)-1])

def k_means_clust(data, num_clust, num_iter, headers):
    print('clustering begins. this will take a few minutes depending on iterations:', num_iter, ' and number of clusters:', num_clust)
    centroids = random.sample(list(data), num_clust)
    counter = 0
    trendVars = None
    for n in range(num_iter):
        trendVars = np.zeros((data.shape[0], num_clust), dtype=float)
        counter+=1
        assignments={}
        #assign data points to clusters
        for ind, i in enumerate(data):
            min_dist=float('inf')
            closest_clust=None
            for c_ind, j in enumerate(centroids):
                cur_dist=euclid_dist_w_missing(i, j) #DTWDistance(i,j,w)
                if cur_dist<min_dist:
                    min_dist=cur_dist
                    closest_clust = c_ind
            trendVars[ind, closest_clust] = 1.0
            if closest_clust in assignments:
                assignments[closest_clust].append(ind)
            else:
                assignments[closest_clust]=[ind]
    
        #recalculate centroids of clusters
        standardDevCentroids = np.zeros((num_clust, data[0].shape[0], data[0].shape[1]), dtype=float)
        for key in assignments:
            clust_sum = np.zeros(data[0].shape, dtype=float)
            clust_cnt = np.zeros(data[0].shape, dtype=float)

            for k in assignments[key]:
                clust_sum += data[k]
                clust_cnt += (data[k] != 0) * 1
            clust_cnt[clust_cnt == 0] = 1
            centroids[key] = clust_sum / clust_cnt 

            if n == num_iter - 1:
                clust_sum=np.zeros(data[0].shape, dtype=float)
                clust_cnt=np.zeros(data[0].shape, dtype=float)
                for k in assignments[key]:
                    nonzeroix = ((data[k]!=0) & (centroids[key] != 0)) * 1.0
                    clust_sum += (nonzeroix * data[k] - nonzeroix * centroids[key]) ** 2
                    clust_cnt += nonzeroix
                clust_cnt[clust_cnt == 0] = 1.0
                standardDevCentroids[key][:,:] = clust_sum / clust_cnt 

    cnt_clusters = [len(assignments[k]) for k in assignments]
    print("Done! Number of datapoints per cluster is ", cnt_clusters)
    return centroids, assignments, trendVars, standardDevCentroids

def plot_trends(centroids, headers, standardDevCentroids):
    # plot_trends(centroids, headers)
    vital_types = [h.strip('-avg0to3').split(':')[1] for h in headers[0,:]]
    print(vital_types)
    sizex = math.floor(np.sqrt(len(centroids)))
    sizey = math.ceil(np.sqrt(len(centroids)))
    fig, axes = plt.subplots(sizex, sizey)
    for i in range(0, sizex):
        for j in range(0, sizey):
            centroids_ix =  i*sizey + j
            if centroids_ix >= len(centroids):
                break
            for vitalix in range(0, len(vital_types)):
                axes[i,j].plot(centroids[centroids_ix][:,vitalix], label=vital_types[vitalix])
                axes[i,j].fill_between(range(len(centroids[centroids_ix][:,vitalix])), centroids[centroids_ix][:,vitalix]+standardDevCentroids[centroids_ix][:,vitalix], centroids[centroids_ix][:,vitalix]-standardDevCentroids[centroids_ix][:,vitalix], alpha=0.1)
    axes[0,0].legend(fontsize = 6)
    
                
# from: http://nbviewer.jupyter.org/github/alexminnaar/time-series-classification-and-clustering/blob/master/Time%20Series%20Classification%20and%20Clustering.ipynb
import numpy as np
import random
random.seed(1)
import math
try:
    import matplotlib.pyplot as plt
except:
    print('cant plot. install matplotlib if you want to visualize')
import pickle

def load_temporal_data(xtrain, headers, ytrain, ylabels):
    print(headers)
    print(xtrain.shape)
    vital_count = 9
    newh = headers.reshape(vital_count, 11)
    newx = xtrain.reshape(xtrain.shape[0], vital_count, 11)
    pickle.dump(obj=(newx[:,1:,:], newh[1:,:], ytrain, ylabels), file=open('tmpobj_20170811.pkl', 'wb'), protocol=2)
    return newx[:,1:,:], newh[1:,:]

def unpickle_data(fname='tmpobj_20170811.pkl'):
    (newx, newh, ytrain, ylabels) = pickle.load(open(fname, 'rb'))
    return (newx, newh, ytrain, ylabels)

def euclid_dist(t1,t2):
    return np.sqrt(sum((t1-t2)**2))

def euclid_dist_w_missing(t1, t2):
    #[' BMI', ' BMI Percentile', ' Fundal H', ' HC', ' HC Percentile', ' H', ' Ht Percentile', ' Pre-gravid W', ' W', ' Wt Change', ' Wt Percentile']
    subset = [True, False, False, False, False, True, False, False, True, False, False]
    nonzeros = (t1[:,subset] != 0) & (t2[:,subset] != 0)
    if nonzeros.sum() == 0:
        return float('inf')
    return np.sqrt(sum((t1[:,subset][nonzeros]-t2[:,subset][nonzeros])**2))/nonzeros.sum()

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
                if closest_clust == None:
                    closest_clust = 0
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
    return centroids, assignments, trendVars, standardDevCentroids, cnt_clusters

def plot_trends(centroids, headers, standardDevCentroids, cnt_clusters=[]):
    # plot_trends(centroids, headers)
    vital_types = [h.strip('-avg0to3').split(':')[1] for h in headers[0,:]]
    print(vital_types)
    sizex = math.ceil(np.sqrt(len(centroids)))
    sizey = math.ceil(np.sqrt(len(centroids)))
    fig, axes = plt.subplots(sizex, sizey)
    for i in range(0, sizex):
        for j in range(0, sizey):
            centroids_ix =  i*sizey + j
            if centroids_ix >= len(centroids):
                break
            for vitalix in [0, 8]: #range(0, len(vital_types)):
                axes[i,j].plot(centroids[centroids_ix][:,vitalix], label=vital_types[vitalix])
                axes[i,j].set_title('Trend:'+str(centroids_ix) + ' cnt:'+str(cnt_clusters[centroids_ix]))
                axes[i,j].fill_between(range(len(centroids[centroids_ix][:,vitalix])), centroids[centroids_ix][:,vitalix]+standardDevCentroids[centroids_ix][:,vitalix], centroids[centroids_ix][:,vitalix]-standardDevCentroids[centroids_ix][:,vitalix], alpha=0.1)
    axes[0,0].legend(fontsize = 6)
    
                

def build_endtoend_model(x, h, y, ylabels, xtest, ytest, params={'clustercnts':4, 'maxepoch':1000}):
    try:
        import tensorflow as tf
        config = tf.ConfigProto(
            device_count = {'GPU': 0}
        )
        tf.reset_default_graph()
    except:
        print ('tensorflow not loading. Make sure it is installed and can be imported')
        return

    cluster_cnt = params['clustercnts']
    maxepoch = params['maxepoch']
    
    print(x.shape)
    x_input = tf.placeholder(tf.float32, shape=[None, x[0].shape[0], x[0].shape[1]])
    print('initiating network with:', cluster_cnt, ' clusters')
    patterns = []
    for i in range(0, cluster_cnt):
        patterns.append( tf.Variable(tf.zeros([x[0].shape[0], x[0].shape[1]]), name='pattern'+str(i)) )
    # k1 = tf.Variable(tf.zeros([x[0].shape[0], x[0].shape[1]]), name='pattern'+str(0)) 
    
    net_pattern = []
    for p in patterns:
        net_pattern.append(tf.reduce_mean(tf.squared_difference(x_input, p)))
    # k2 = tf.squared_difference(x_input, k1)

    net_d = tf.stack(net_pattern)
    loss = tf.reduce_min(net_d)

    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    sess = tf.InteractiveSession()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    averages = []
    for ep in range(0, maxepoch):
        losslist = []
        for i in range(0,len(x)):
            out = sess.run([train_step, loss, patterns], feed_dict={x_input:x[i].reshape(-1, x[i].shape[0],x[i].shape[1])})
            # print(out[2][0])
            losslist.append(out[1])
        
        averages.append(sum(losslist)/len(losslist))
        plt.plot(averages)
        plt.draw()
        
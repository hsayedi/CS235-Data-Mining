"""
Husna Sayedi
Code for CS-235 Final Project
"""


# Full K-Means Algorithm

'''
This function will take in a 2-dim pandas DataFrame and run a K-Means Algorithm on it.
Inputs required are: DataFrame, number of iterations (after convergence KMeans will give same solution),
and the number of clusters (K). K often requires intuition based on the dataset. 
The PlotElbow() function can help to determine an optimal number for K. 
'''

def KMeansAlgorithm(df, num_iter, K):
    
    # imports
    import numpy as np
    import pandas as pd
    import random as rd
    import seaborn as sns
    from matplotlib import pyplot as plt

    data = df.values 
    m = data.shape[0] # num training examples
    n = data.shape[1] # num of features
    
    def InitCentroidsRandom():
        # Centroids will be a (n x K) dimensional matrix. Each column will be one centroid for one cluster
        centroids = np.array([]).reshape(n, 0)
        for i in range(K): 
            rand = rd.randint(0, m-1)
            centroids = np.c_[centroids,data[rand]]
        return centroids # the KMeansAlgorithm() function will return centroids.T
    
    # Initiate centroids randomly 
    centroids = InitCentroidsRandom()
    result = {}
    
    # Begin iterations to update centroids, compute and update Euclidean distances
    for i in range(num_iter):
         # First compute the Euclidean distances and store them in array
          EucDist = np.array([]).reshape(m, 0)
          for k in range(K):
              dist = np.sum((data - centroids[:,k])**2, axis=1)
              EucDist = np.c_[EucDist, dist]
          # take the min distance 
          min_dist = np.argmin(EucDist, axis=1) + 1 
            
         # Begin iterations
          soln_temp = {} # temp dict which stores solution for one iteration - Y
            
          for k in range(K):
              soln_temp[k+1] = np.array([]).reshape(n, 0)
           
          for i in range(m):
              # regroup the data points based on the cluster index 
              soln_temp[min_dist[i]] = np.c_[soln_temp[min_dist[i]], data[i]]
          
          for k in range(K):
              soln_temp[k+1] = soln_temp[k+1].T
          # Updating centroids as the new mean for each cluster
          for k in range(K):
              centroids[:,k] = np.mean(soln_temp[k+1], axis=0)
          result = soln_temp
        
    def PlotClusters(result, centroids, K):
        # create arrays for colors and labels based on specified K
        colors = ["#"+''.join([rd.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(K)]
        labels = ['cluster_' + str(i+1) for i in range(K)]
        
        fig1 = plt.figure(figsize=(5,5))
        ax1 = plt.subplot(111)
        # plot each cluster
        for k in range(K):
            ax1.scatter(result[k+1][:,0], result[k+1][:,1], 
                        c = colors[k], label = labels[k])
        # plot centroids
        ax1.scatter(centroids[0,:],centroids[1,:], #alpha=.5,
                    s = 300, c = 'lime', label = 'centroids')
        plt.xlabel(df.columns[0]) # first column of df
        plt.ylabel(df.columns[1]) # second column of df
        plt.legend()
        plt.show()

        return 
            
    return result, centroids.T, PlotClusters(result, centroids, K)

'''
Elbow Method:
The elbow method will help us determine the optimal value for K. 
Steps: 
1) Use a range of K values to test which is optimal 
2) For each K value, calculate Within-Cluster-Sum-of-Squares (WCSS) 
3) Plot Num Clusters (K) x WCSS
'''
def PlotElbow(df, n_iter, K):
    
    wcss_vals = np.array([])
    for k_val in range(1,K):
#         print("K = {}".format(k_val))
        results = KMeansAlgorithm(df, n_iter, k_val)
        output = results[0]
        centroids = results[1]
#         print('Centroids:')
#         print(centroids)
        wcss=0
        for k in range(k_val):
            wcss += np.sum((output[k+1] - centroids[k,:])**2)
        wcss_vals = np.append(wcss_vals, wcss)
    # Plot K values vs WCSS values
    K_vals = np.arange(1, K)
    plt.plot(K_vals, wcss_vals)
    plt.xlabel('K Values')
    plt.ylabel('WCSS')
    plt.title('Elbow Method')
    plt.show()

'''
Function to remove large outliers and noise in pandas DataFrames
Use before running the KMeansAlgorithm
'''
def remove_outliers(df):
    low = .10 #.05
    high = .90 #.95
    quant_df = df.quantile([low, high])
    for name in list(df.columns):
      if ptypes.is_numeric_dtype(df[name]):
       df = df[(df[name] > quant_df.loc[low, name]) 
           & (df[name] < quant_df.loc[high, name])]
    return df



'''
Create pandas Series to run through Algorithm as a sample data set

We are interested in seeing if there is a correlation between 
price_normalized and number_of_reviews. The number of reviews
can be seen as a feature that shows a listing's popularity. 
We create separate Series for each room type, then further 
slice the data by neighbourhood_group 

We always use our above function remove_outliers()
since K-Means algorithm is very sensitive to noise and outliers
'''
def generate_data(path):
	df = pd.read_csv('data/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
	# Entire Home / Apartment - Listings
	home = df[df.room_type == 'Entire home/apt']
	home_staten = home_staten[['price_normalized', 'number_of_reviews']]
	home_staten = remove_outliers(home_staten)   

	return home_staten













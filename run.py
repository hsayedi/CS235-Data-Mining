
import sys
import pandas as pd
from KMeans import generate_data, KMeansAlgorithm, PlotElbow
from Neural_Network import gen_graphs


path = 'data/new-york-city-airbnb-open-data/AB_NYC_2019.csv'

df_kmeans = generate_data(path)

print("1 - Plotting the Elbow Method with iterations=100 and K=10 clusters...")
PlotElbow(df_kmeans, 100, 10)

print()
print("2 - Running KMeansAlgorithm now with K=4 clusters...")
KMeansAlgorithm(df_kmeans, 100, 4)

print()
print("3 - Running Neural Network now with 1 hidden layer, 8 neurons in layer")
gen_graphs(path, list_of_neurons=[8], epochs=10) # default list_of_neurons=[8, 16, 32], epochs=100 but can take a long time to run

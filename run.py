
import sys
import pandas as pd
from KMeans import generate_data, KMeansAlgorithm, PlotElbow


path = 'data/new-york-city-airbnb-open-data/AB_NYC_2019.csv'

df_kmeans = generate_data(path)
print("Starting Implementation #1: K-Means Clustering...")
print()
print()
print("Step 1 - Plotting the Elbow Method with iterations=100 and K=10 clusters...")
PlotElbow(df_kmeans, 100, 6)

print()
print("Step 2 - Running KMeansAlgorithm now with K=4 clusters...")
KMeansAlgorithm(df_kmeans, 100, 4)
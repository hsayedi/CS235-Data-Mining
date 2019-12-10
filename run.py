
import sys
import pandas as pd
from KMeans import generate_data, KMeansAlgorithm, PlotElbow


path = 'data/new-york-city-airbnb-open-data/AB_NYC_2019.csv'

df_kmeans = generate_data(path)

print("1 - Plotting the Elbow Method with iterations=100 and K=10 clusters...")
PlotElbow(df_kmeans, 100, 10)

print()
print("2 - Running KMeansAlgorithm now with K=4 clusters...")
KMeansAlgorithm(df_kmeans, 100, 4)
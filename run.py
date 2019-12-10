
import sys
from KMeans import generate_data, KMeansAlgorithm, PlotElbow


path = ''

df_kmeans = generate_data(path)

PlotElbow(df_kmeans, 100, 10)
KMeansAlgorithm(df_kmeans)
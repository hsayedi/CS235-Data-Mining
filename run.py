
import sys
from KMeans import generate_data, KMeansAlgorithm, PlotElbow
from Neural_Network import process_data, create_model, test_models, plot_histories, gen_graphs


path = ''

df_kmeans = generate_data(path)

PlotElbow(df_kmeans, 100, 10)
KMeansAlgorithm(df_kmeans)

gen_graphs(path, list_of_neurons=[8], epochs=10) # default list_of_neurons=[8, 16, 32], epochs=100 but can take a long time to run
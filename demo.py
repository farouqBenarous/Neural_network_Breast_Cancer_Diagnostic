import pandas as pd              # A beautiful library to help us work with data as tables
import numpy as np               # So we can use number matrices. Both pandas and TensorFlow need it. 
import matplotlib.pyplot as plt  # Visualize the things
import tensorflow as tf          # Fire from the gods

if __name__ == "__main__":

    # Let's have Pandas load our dataset as a dataframe
    dataframe = pd.read_csv("datasetcsv.csv")
    # remove columns we don't care about
    dataframe = dataframe.drop(["unknown.17","unknown.16","unknown.15","unknown.14","unknown.13","unknown.12",
                                "unknown.11","unknown.10","unknown.9","unknown.8","unknown.7","unknown.6","unknown.5"
                                   ,"unknown.4","unknown.3","unknown.2","unknown.1","unknown"],
                                axis=1)
    # We'll only use the first 10 rows of the dataset in this example
    dataframe = dataframe[0:10]
    #Let's have the notebook show us how the dataframe looks now
    print(dataframe)
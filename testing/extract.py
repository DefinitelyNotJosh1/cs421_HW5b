# extract weights and turn them into a numpy array in a text file - so I can copy/paste weights and biases

import numpy as np
import os
import pandas as pd



# load the weights and biases from the npz file
weights_and_biases_file = "testing/weights10_10_20_OLD.npz"
with open(weights_and_biases_file, "rb") as f:
    data = np.load(f)
    w1 = data["w1"]
    b1 = data["b1"]
    w2 = data["w2"]
    b2 = data["b2"]
    w3 = data["w3"]
    b3 = data["b3"]

# print the weights and biases to a text file in the format of a numpy array
with open("testing/extracted_weights.txt", "a") as f:
    f.write(f"w1 = {w1}\n")
    f.write(f"b1 = {b1}\n")
    f.write(f"w2 = {w2}\n")
    f.write(f"b2 = {b2}\n")
    f.write(f"w3 = {w3}\n")
    f.write(f"b3 = {b3}\n")

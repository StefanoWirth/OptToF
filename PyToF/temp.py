
import os
import numpy as np

all_arrays = []

for filename in os.listdir("PyToF/BaseRhos/"):
    file_path = os.path.join("PyToF/BaseRhos/", filename)
    array = np.load(file_path)
    all_arrays.append(array)

concatenated_array = np.stack(all_arrays, axis=0)

np.save("PyToF/baserhosuranus.npy", concatenated_array)
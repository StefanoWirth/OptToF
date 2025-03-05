import numpy as np
import scipy as sp
from math import comb
import random
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
#import functools

sys.setrecursionlimit(100000)

Resolution = 10 # resolution of density curve
Layers = 7 # number of layers

R = 2**Resolution
N = 2**Layers

ResultFunction = np.zeros(N)


Number_of_functions = comb(N + R - 2, N - 1) 


#random.seed(1)


Balls2Place = R - 1
BinTotal = N
BinID = 0
Balls2Bin = Balls2Place


#@functools.cache
def tryplacemax(func_id, Balls2Place, Balls2Bin, BinID):

    if func_id < 0 or Balls2Place < 0 or Balls2Bin < 0 or BinID >= BinTotal:
        return
    
    if BinID == BinTotal - 1:
        ResultFunction[BinID] = Balls2Bin
        return

    Balls2OtherBins = Balls2Place - Balls2Bin
    NrOfOtherBins = BinTotal - BinID - 1
    ways2place = comb(Balls2OtherBins + NrOfOtherBins - 1, NrOfOtherBins - 1)
    #print("Current function state:")
    #print(ResultFunction)
    #print("Trying to place " + str(Balls2Bin) + " Balls into bin number " + str(BinID))

    if func_id < ways2place:
        ResultFunction[BinID] = Balls2Bin
        tryplacemax(func_id, Balls2OtherBins, Balls2OtherBins, BinID + 1)

    elif func_id >= ways2place:
        tryplacemax(func_id - ways2place, Balls2Place, Balls2Bin - 1, BinID)

    else:
        raise Exception("shit")



plt.subplot(2, 2, 1)
print("Uniformly Chosen generator:")
for i in tqdm(range(10)):
    func_id = random.randint(0,Number_of_functions-1)
    #func_id = int(Number_of_functions * i / 5)
    #print("Random value is: " + str(func_id))
    tryplacemax(func_id, Balls2Place, Balls2Bin,BinID)
    for i in range(N-1):
        ResultFunction[i+1] = ResultFunction[i+1]+ResultFunction[i]
    ResultFunction = ResultFunction / (R - 1)
    #print(ResultFunction)
    plt.plot(ResultFunction)

plt.subplot(2, 2, 2)
print("Random step generator:")
for i in tqdm(range(20)):
    ResultFunction = np.zeros(N)

    for i in range (N-1):
        ResultFunction[i+1] = random.random()*2/N+ResultFunction[i]
    scale = 1/ResultFunction[-1]
    ResultFunction = ResultFunction * scale
    plt.plot(ResultFunction)


#lower inclusive, upper inclusive 
def subdivide(lower_x, upper_x, lower_y, upper_y):
    if upper_x == lower_x:
        ResultFunction[lower_x] = random.uniform(lower_y,upper_y)
        return
    x = random.randint(lower_x ,upper_x)
    y = random.uniform(lower_y,upper_y)
    ResultFunction[x] = y
    if lower_x < x:
        subdivide(lower_x, x - 1, lower_y, y)
    if x < upper_x:
        subdivide(x + 1, upper_x, y, upper_y)

plt.subplot(2, 2, 3)
print("Subdivide generator:")
for i in tqdm(range(200)):
    ResultFunction = np.zeros(N)
    ResultFunction[-1] = 1
    subdivide(1,N - 2, 0, 1)
    plt.plot(ResultFunction)


max_depth = Layers

#lower inclusive, upper inclusive 
def subdivide_binary(pivot, depth, lower_y, upper_y):
    y = random.uniform(lower_y,upper_y)
    ResultFunction[pivot] = y
    depth = depth + 1
    pivot_step = int(2**(max_depth-depth-1))
    if pivot_step == 0:
        return
    subdivide_binary(pivot-pivot_step, depth, lower_y, y)
    subdivide_binary(pivot+pivot_step, depth, y, upper_y)


plt.subplot(2, 2, 4)
print("Subdivide Binary generator:")
for i in tqdm(range(200)):
    ResultFunction = np.zeros(N)
    ResultFunction[-1] = 1
    subdivide_binary(64,0, 0, 1)
    plt.plot(ResultFunction)



plt.show()

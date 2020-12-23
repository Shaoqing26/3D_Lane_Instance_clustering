import argparse
from pypcd import pypcd
import numpy as np
import open3d
import random
import math
import pandas as pd
# import geopandas as gp
import matplotlib.pyplot as plt
from sklearn import linear_model
from itertools import zip_longest
# list_1=[1,2,3,4,5,6]
# def double_func(x):
#     return(x*2)
# list_2=map([],list_1)
# print(list(list_2))

# gt_ = np.load('data_.npy',allow_pickle=True).item()
a = [[[[1, 1, 0]], [[2, 2, 0]],[2,2,3]],
     [[[0, 2, 0]], [[2, 1, 1], [2, 2, 1]]],
     [[[0, 2, 0]], [[2, 1, 1]]]]
b = [[np.array([[1, 1, 0]], [[2, 2, 0]],[[2,2,3]])],
     [np.array([[0, 2, 0], [2, 1, 1], [2, 2, 1]])],
     [np.array([[0, 2, 0], [2, 1, 1]])]]
# print(list(map([],a)))

cat_list = [np.concatenate(i) for i in zip_longest(*a,fillvalue=[])]
cat_list = [np.concatenate(i) for i in zip(*a)]
# map([],a)
print(a)

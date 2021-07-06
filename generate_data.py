import numpy as np
import os
import matplotlib.pyplot as plt
from numpy.lib.shape_base import expand_dims

def generate_list():
    L = []
    f = open('figurelist.txt', 'w')
    for root, dir, files in os.walk('npy'):
        for file in files:
            if os.path.splitext(file)[1] == '.npy':
                f.write(os.path.join(root, file))
                f.write('\n')
                L.append(os.path.join(root, file))
    f.close()
    return L

def Append(X, Y):
    if X is None:
        return Y
    else:
        return np.concatenate((X, Y), axis = 0)


files_path = generate_list()
f = open('evallist.txt', 'w')

Origin = None
for file_path in files_path:
    src = np.load(file_path)
    arr = list(np.hstack(src))
    pre = arr.count(0)
    if pre < 0.35 * len(arr):
        src = expand_dims(src, axis = 0)
        Origin = Append(Origin, src)
        print(Origin.shape)
    else:
        if pre < 0.4 * len(arr):
            f.write(file_path)
            f.write('\n')
        
np.save('origin.npy', Origin)
print(Origin.shape)
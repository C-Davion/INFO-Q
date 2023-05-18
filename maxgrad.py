import numpy as np
import matplotlib.pyplot as plt
from generate import degree

def maxgrad(l): #trouve l'epoch de gradiant maximal (terme de droite) dans une liste de la forme  [(epoch,loss)]
    hold=l[0][0]
    grad=-np.inf
    for i in range (len(l)-1):
        delta=l[i][1]-l[i+1][1]
        if delta>grad:
            grad=delta
            hold=l[i+1][0]
    return hold



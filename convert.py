from generate import degree
import numpy as np

def shaping(data):
    for i in range(0,len(data),degree): #transfrom the csv into the suited format (csv stores arrays as string ffs)
        input_vectors=[]
        outputs_vectors=[]
        if i + degree <= len(data):
            input_vectors.append(data[i:i+degree, :2])
            data[i,2]=data[i,2].replace("\n","")
            data[i,2]=data[i,2].replace("[","")
            data[i,2]=data[i,2].replace("]","")
            outputs_vectors.append(np.fromstring(data[i,2],sep=" "))
    return np.array(input_vectors),np.array(outputs_vectors)

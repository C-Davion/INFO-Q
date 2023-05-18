import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from generate import degree
from maxgrad import maxgrad
from convert import shaping

'''
Code tenu par des bout de scotch.

Data generated in generate.py and saved into a csv

Input:
Tensor of size (degree 2) containit on the first column the x, and on the second one f(x) 

Output:
Tensor of size(degree+1) containing the coefficient of the polynomials.

A small code to compute the maximum gradient of the loss function is added. Details in the convert file.
'''


df=pd.read_csv('deg6_100.csv') #read the data, make sure to input the correct file

data=df[['x','p(x)','polynome']].values

input_vectors=[]
outputs=[]
num_lines=degree
epochs=10 #why tf not

(input_vectors,outputs)=shaping(data)

input_tensors = tf.convert_to_tensor(input_vectors, dtype=tf.float32)
output_tensors = tf.convert_to_tensor(outputs, dtype=tf.float32)

print(output_tensors[0])

loss_values=[] #placeholder for the loss function

model=tf.keras.models.Sequential([ #avg tf code
    tf.keras.layers.InputLayer(input_shape=input_tensors[0].shape),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(1)
])  

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss='mean_squared_error')

for epoch in range(epochs):
    print(f'epoch={epoch}') #track progress
    model.fit(input_tensors, output_tensors, batch_size=2) #small batch_size bc we are fitting polynoimals, it's going to overfitt af
    
    # Compute the loss for this epoch
    loss = model.evaluate(input_tensors, output_tensors)
    loss_values.append((epoch, loss))

loss_on_epoch=[x[1] for x in loss_values]
epoch=[x[0] for x in loss_values]
plt.plot(epoch,loss_on_epoch,'b-o')
print(f'the maximum gradiant occured at {maxgrad(loss_values)}' )
plt.show()

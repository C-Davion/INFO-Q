import numpy as np
import matplotlib.pyplot as plt

# Load the data from the file 
data = np.loadtxt('generated_data\NNloss.txt')
databis=np.loadtxt('generated_data\TN8einloss.txt')

# Extract columns
start=100
filtered_data = data[start::100]
filbis=databis[start::100]
x = filtered_data[:, 0]  # First column
y = filtered_data[:, 1]  # Second column
xbis=filbis[:,0]
ybis=filbis[:,1]

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(x, y,  linestyle='-', color='b', label='NNLoss')
plt.plot(xbis,ybis,linestyle='-',color='r',label='EinLoss')

# Add labels and title
plt.xlabel('N iterations')
plt.ylabel('Loss')
plt.title('Loss over N_iterations')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
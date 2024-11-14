import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt


def relecart(Pred,ref):
    '''
    compute the relative difference of two vector of same size w/ the euclidian norm
    ''' 
    try:
        assert np.shape(Pred)==np.shape(ref),"Not the same shape"
        nref=norm(ref)
        return norm(Pred-ref)/nref if nref !=0 else norm(Pred-ref)
    except AssertionError as e:
        return str(e)
    

def avgecart(Pred,ref):
    '''
    Input: Pref and ref two arrays of same length where each elements is a vector
    Output: The average relative difference element wize 
    '''
    try:
        assert np.shape(Pred)[0]==np.shape(ref)[0], "Not the same lenght"
        temp=relecart(Pred,ref)
        return np.average(temp)

    except AssertionError as e:
        return str(e)
    

file_name="tn8eindatanew100_0.001_20000.npz"
file_namebis='tndata100_0.001_20000.npz'

datatn = np.load(file_name, allow_pickle=True)
datann=np.load(file_namebis,allow_pickle=True)

# Access the arrays
Xtnpred=datatn['X_pred']
Ytnpred=datatn['Y_pred']
Ytntest=datatn['Y_test']
ttntest=datatn['t_test']
#load the nn data
Xnnpred=datann['X_pred']
Ynnpred=datann['Y_pred']
Ynntest=datann['Y_test']
tnntest=datann['t_test'] 
# Print the first element of X_pred_loaded
#print("First element of X_pred_loaded:", np.shape(Ytest[0]))
#print((Ynnpred[0]))
print(avgecart(Ynnpred,Ynntest))
samples=5
M = 100  # number of trajectories (batch size)
N = 50  # number of time snapshots
D = 100  # number of dimensions
T = 1.0
plt.figure()
plt.plot(ttntest[0:1,:,0].T,Ytnpred[0:1,:,0].T,'b',label='Learned by MPO $u(t,X_t)$')
plt.plot(ttntest[0:1,:,0].T,Ytntest[0:1,:,0].T,'r--',label='Exact $u(t,X_t)$')
plt.plot(ttntest[0:1,-1,0],Ytntest[0:1,-1,0],'ko',label='$Y_T = u(T,X_T)$')
plt.plot(ttntest[0:1,-1,0],Ynntest[0:1,-1,0],'ko',label='$Y_T = u(T,X_T)$')
plt.plot(ttntest[0:1,:,0].T,Ynnpred[0:1,:,0].T,'g',label='Learned by MPO64 $u(t,X_t)$')

plt.plot(ttntest[1:samples,:,0].T,Ytnpred[1:samples,:,0].T,'b')
plt.plot(ttntest[1:samples,:,0].T,Ytntest[1:samples,:,0].T,'r--')
plt.plot(ttntest[1:samples,-1,0],Ytntest[1:samples,-1,0],'ko')
plt.plot(ttntest[1:samples,-1,0],Ynntest[1:samples,-1,0],'ko')
plt.plot(ttntest[1:samples,:,0].T,Ynnpred[1:samples,:,0].T,'g')


plt.plot([0],Ytntest[0,0,0],'ks',label='$Y_0 = u(0,X_0)$')

plt.xlabel('$t$')
plt.ylabel('$Y_t = u(t,X_t)$')
plt.title('100-dimensional Black-Scholes-Barenblatt')
plt.legend(ncol=2)
plt.show()


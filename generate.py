import pandas as pd
import numpy as np

degree=6 # degree du polynome
nombre=100# nombre de polynome différent.


def creerpoly():
    return np.add(np.random.rand(degree+1),np.random.randint(100,size=(degree+1)))


def abscisse():
    return np.random.choice(np.arange(0, 1, 0.001), size=degree, replace=False)



def evaluate(P,x):
    sum=0
    for i in range(degree+1):
        sum+=P[i]*x**i
    return sum

def createdf():
    P=creerpoly() #change it to creerpoly to have a random one
    x=abscisse()
    fx=[evaluate(P,y) for y in x]
    ans=[P]*degree
    frame=pd.DataFrame({'x':x, 'p(x)':fx,'polynome':ans })
    return frame

#join the dataframe multiple times to get


def createset():
    res=createdf()
    filename=f'deg{degree}_{nombre}.csv'
    for i in range(nombre):
        temp=createdf()
        res=pd.concat([res,temp],axis=0,ignore_index=True)
    res.to_csv(filename,index=False)


#createset() #uncomment it and run it to generate the csv and then comment it again.





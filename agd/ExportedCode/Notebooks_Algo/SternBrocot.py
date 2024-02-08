# Code automatically exported from notebook SternBrocot.ipynb in directory Notebooks_Algo
# Do not modify
import numpy as np
from matplotlib import pyplot as plt

def MakeStencil(refine_pred):
    l = [np.array([1,0]),np.array([0,-1]),np.array([-1,0]),np.array([0,1])]
    m = [np.array([1,0])]
    while len(l)>0:
        u=m[-1]
        v=l[-1]
        if(refine_pred(u,v)):
            l.append(u+v)
        else:
            m.append(v)
            l.pop()
    return m

def PlotStencil(stencil):
    plt.plot(*np.array(stencil).T)
    plt.scatter(*np.array(stencil).T)
    plt.scatter(0,0,color='black')

aX0 = np.linspace(-1,1); aX1=aX0
X = np.array(np.meshgrid(aX0,aX1,indexing='ij'))

def ball_and_stencil(metric,level,name):
    plt.figure(figsize=[12,4])
    plt.subplot(1,2,1); plt.title("Unit ball for a norm of "+name+" type"); plt.axis('equal')
    plt.contourf(*X,metric.norm(X),levels=[0.,level]); plt.scatter(0,0,color='black'); 
    plt.subplot(1,2,2); plt.title("Stencil for a norm of "+name+" type"); plt.axis('equal')
    PlotStencil(MakeStencil(lambda u,v: metric.angle(u,v)>np.pi/3))


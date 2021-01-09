# Code automatically exported from notebook ShapeFromShading.ipynb in directory Notebooks_NonDiv
# Do not modify
import numpy as np
import matplotlib.pyplot as plt

def EvalScheme(cp,u,uc,params):
    """
    Evaluates the (piecewise) quadratic equation defining the numerical scheme.
    Inputs :
     - uc : plays the role of λ
    """
    alpha,beta,gamma,h = params
    wx = np.roll(u,-1,axis=0)
    wy = np.roll(u,-1,axis=1)
    vx = np.minimum(wx,np.roll(u,1,axis=0))
    vy = np.minimum(wy,np.roll(u,1,axis=1))

    return (cp*np.sqrt(1+(np.maximum(0,uc-vx)**2+np.maximum(0,uc-vy)**2)/h**2) +
            alpha*(uc-wx)/h+beta*(uc-wy)/h-gamma)

def LocalSolve(cp,vx,vy,wx,wy,params):
    """
    Solve the (piecewise) quadratic equation defining the numerical scheme.
    Output: solution λ.
    """
    alpha,beta,gamma,h = params
    # Trying with two active positive parts 
    
    # Quadratic equation coefficients.
    # a lambda^2 - 2 b lambda + c =0
    a = (2.*cp**2 - (alpha+beta)**2)
    b = cp**2 *(vx+vy) - (alpha+beta)*(alpha*wx+beta*wy+h*gamma)
    c = cp**2*(h**2+vx**2+vy**2)-(gamma*h+alpha*wx+beta*wy)**2
    
    delta = b**2 - a*c
    good = np.logical_and(delta>=0,a!=0)
    u = 0*cp;
    # TODO : Is that the correct root selection ?
    u[good] = (b[good]+np.sqrt(delta[good]))/a[good] 

    vmax = np.maximum(vx,vy)
    good = np.logical_and(good,u>=vmax)
    
    # Trying with one active positive part
    # TODO : restrict computations to not good points to save cpu time ?
    
    vmin = np.minimum(vx,vy)
    a = (cp**2 - (alpha+beta)**2)
    b = cp**2 *vmin - (alpha+beta)*(alpha*wx+beta*wy+h*gamma)
    c = cp**2*(h**2+vmin**2)-(gamma*h+alpha*wx+beta*wy)**2

    delta = b**2 - a*c
    ggood = np.logical_and(np.logical_and(delta>=0,a!=0), 1-good)
    u[ggood] = (b[ggood] +np.sqrt(delta[ggood]))/a[ggood]
    
    good = np.logical_or(good,np.logical_and(ggood,u>=vmin))
    
    # No active positive part
    # equation becomes linear, a lambda - b = 0
    a = alpha+beta+0.*cp
    b = alpha*wx+beta*wy +gamma*h - cp*h
    bad = np.logical_not(good)
    u[bad]=b[bad]/a[bad]
    return u
    
def JacobiIteration(u,Omega,c,params):
    """
    One Jacobi iteration, returning the pointwise solution λ to the numerical scheme.
    """
    wx = np.roll(u,-1,axis=0)
    wy = np.roll(u,-1,axis=1)
    vx = np.minimum(wx,np.roll(u,1,axis=0))
    vy = np.minimum(wy,np.roll(u,1,axis=1))
    
#    sol=LocalSolve(c,vx,vy,wx,wy,params)
    sol = u+LocalSolve(c,vx-u,vy-u,wx-u,wy-u,params)
    u[Omega] = sol[Omega]

def OneBump(x,y):
    bump = 0.5-3.*((x-0.5)**2+(y-0.5)**2)
    return np.maximum(bump, np.zeros_like(x))

def ThreeBumps(x,y):
    bump1 = 0.3-3*((x-0.4)**2+(y-0.5)**2)
    bump2 = 0.25-3*((x-0.65)**2+(y-0.6)**2)
    bump3 = 0.25-3*((x-0.6)**2+(y-0.35)**2)
    return np.maximum.reduce([bump1,bump2,bump3,np.zeros_like(bump1)])

def Volcano(x,y):
    r = np.sqrt((x-0.5)**2+(y-0.5)**2)
    volcano = 0.05+1.5*(1+x)*(r**2-6*r**4)
    return np.maximum(volcano, np.zeros_like(x))

def GenerateRHS(height,params):
    α,β,γ,h = params
    hx,hy = np.gradient(height,h)
    Intensity = (α*hx+β*hy+γ)/np.sqrt(1+hx**2+hy**2)
    Omega = height>0
    return Intensity,Omega


# Code automatically exported from notebook ElasticComparisons.ipynb in directory Notebooks_Div
# Do not modify
from agd.Metrics.Seismic import Hooke
from agd.Metrics import Riemann
from ... import FiniteDifferences as fd
from ... import LinearParallel as lp
from agd.ExportedCode.Notebooks_Div.HighOrderWaves import make_domain,dispersion_e,dispersion_a,MakeRandomTensor
from agd.Eikonal.HFM_CUDA import AnisotropicWave
from agd.ODE.hamiltonian import QuadraticHamiltonian
from ... import AutomaticDifferentiation as ad
from agd.Metrics import Seismic
from agd.Metrics.misc import flatten_symmetric_matrix
from agd.Plotting import savefig; #savefig.dirName = 'Images/ElasticComparisons/'

import numpy as np
import itertools
import copy
from scipy.optimize import linprog
π = np.pi

def shifted_grids(X,dx):
    """
    Shifted grids for staggered scheme.
    X_0,X_1
    X_00,X_10,X_01,X_11 (inverted binary order)
    X_000,X_100,X_010,X_110,X_001,X_101,X_011,X_111
    """
    vdim = len(X)
    shape = X[0].shape
    shifts = itertools.product((0,dx/2),repeat=len(X))
    return [X + fd.as_field(s[::-1],shape) for s in shifts]

class staggered:
    """A class for first order finite difference operators on staggered grids"""
    
    def __init__(self,dx,order=2,bc='Periodic'):
        self.idx=1/dx
        self.order=order; assert order in (2,4)
        self.bc=bc; assert bc in ('Periodic','Dirichlet')

    def roll(self,q,shift,axis,bc=None):
        """Rolls specified axis, inserts zeros in case of Dirichlet boundary conditions"""
        if bc is None: bc=self.bc
        q = np.roll(q,shift,axis)
        if bc!='Periodic': 
            pad = q[*(slice(None),)*axis,shift,None] if bc=='Constant' else 0.
            if shift>0: q[*(slice(None),)*axis,:shift]=pad
            else: q[*(slice(None),)*axis,shift:]=pad
        return q
        
    def diff_left(self,q,axis,s=0): 
        """
        First order finite difference, to the left, along axis i.
        (Result is shifted by half a grid step to the left.)
        """
        dq = self.roll(q,-s,axis)-self.roll(q,1-s,axis) # Centered finite difference, with step 1/2
        if self.order==2: return dq * self.idx 
        dq2 = self.roll(q,-1-s,axis)-self.roll(q,2-s,axis) # Centered finite difference, with step 3/2
        return ( (9/8)*dq-(1/24)*dq2 )*self.idx # Fourth order finite difference
    
    def diff_right(self,q,axis):
        """
        First order finite difference, to the right, along axis i.
        (Result is shifted by half a grid step to the right.)
        """
        return self.diff_left(q,axis,s=1)

    def avg_left(self,q,axis,s=0,bc=None):
        """
        Approximate value, half a grid step to the left, along axis i.
        (Use bc='Constant' for averaging coefficients with Dirichlet boundary conditions.)
        """
        if isinstance(axis,tuple): # Average over several axes
            for ax in axis: q = self.avg_left(q,ax,s,bc)
            return q
        if bc is None: bc = self.bc
        aq = self.roll(q,-s,axis,bc)+self.roll(q,1-s,axis,bc) # Centered average, with step 1/2
        if self.order==2: return aq * 0.5 
        aq2 = self.roll(q,-1-s,axis,bc)+self.roll(q,2-s,axis,bc) # Centered average, with step 3/2
        return (9/16)*aq - (1/16)*aq2 # Fourth order accurate interpolation 
    
    def avg_right(self,q,axis):
        return self.avg_left(q,axis,s=1)

    def diff_left_offset(self,q,axis,offset,right=False):
        """
        First order finite difference, along given offset. Shifts result by 1/2 along axis i to the left.
        - right (bool, optional) : shift to the right instead
        Assumption : offset[axis] must be odd.
        Example : diff_left_offset(q,i,eye[i]) == diff_left(q,i)
        """
        e_left = np.array(offset).astype(int)//2
        e_right = e_left.copy() 
        if np.ndim(right)==0: e_left[axis]+=1-right; e_right[axis]+=right
        else: e_left[axis][np.logical_not(right)]+=1; e_right[axis][right]+=1
        assert np.all(offset==e_left+e_right)
        vdim = len(offset)
        ax = tuple(range(vdim))
        dq = self.roll(q,-e_right,ax) - self.roll(q,e_left,ax)
        if self.order==2: return dq * self.idx # Second order finite difference
        dq2 = self.roll(q,-e_right-offset,ax) - self.roll(q,e_left+offset,ax)
        return ( (9/8)*dq-(1/24)*dq2 )*self.idx # Fourth order finite difference

    def diff_right_offset(self,q,axis,offset):
        """
        Assumption : offset[axis] must be odd.
        Example : diff_right_offset(q,i,eye[i]) == diff_right(q,i)
        """
        return self.diff_left_offset(q,axis,offset,right=True)

def eval_Virieux(qfun,pfun,X,dx,t,dt):
    """
    Evaluate position and momentum at the given position and time, 
    taking into account spatial and temporal grid shifts
    """
    vdim = len(X)
    t2 = t+dt/2
    if vdim==1:
        _,X_1 = shifted_grids(X,dx)
        q0_1 = qfun(t2,X_1)[0]
        p0_1 = pfun(t, X_1)[0]
        return (q0_1,),(p0_1,)
    if vdim==2:
        _,X_10,X_01,_ = shifted_grids(X,dx)
        q0_10 = qfun(t2,X_10)[0]
        q1_01 = qfun(t2,X_01)[1]
        p0_10 = pfun(t, X_10)[0]
        p1_01 = pfun(t, X_01)[1]
        return (q0_10,q1_01),(p0_10,p1_01)
    if vdim==3:
        X_000,X_100,X_010,X_110,X_001,X_101,X_011,X_111 = shifted_grids(X,dx)
        q0_100 = qfun(t2,X_100)[0]
        q1_010 = qfun(t2,X_010)[1]
        q2_001 = qfun(t2,X_001)[2]
        p0_100 = pfun(t, X_100)[0]
        p1_010 = pfun(t, X_010)[1]
        p2_001 = pfun(t, X_001)[2]
        return (q0_100,q1_010,q2_001),(p0_100,p1_010,p2_001)

class Virieux2:
    """The two dimensional Virieux scheme."""
    def __init__(self,ρ,C,stag):
        """
        Inputs : 
        - ρ : density, everywhere positive. 
        - C : Hooke tensor, assumes a VTI structure. Assumes C02=C12=0 and C symmetric.
        - stag : staggered grid difference operators
        """
        self.stag = stag
        ar = stag.avg_right
        # Variable locations on the grid shown after subscript
        iρ = 1/ρ
        self.iρ_10 = ar(iρ,0)
        self.iρ_01 = ar(iρ,1) 
        self.C00_00 = C[0,0]
        self.C01_00 = C[0,1]
        self.C11_00 = C[1,1] 
        self.C22_11 = ar(C[2,2],axis=(0,1))

    def step(self,q,p,dt):
        """
        Inputs : 
        - q,p : position and momentum.
        - dt : time step.
        """
        # Variable locations shown after subscript
        dl,dr = self.stag.diff_left, self.stag.diff_right
        C00_00,C01_00,C10_00,C11_00,C22_11,iρ_10,iρ_01 = self.C00_00,self.C01_00,self.C01_00,self.C11_00,self.C22_11,self.iρ_10,self.iρ_01

        q0_10,q1_01 = copy.deepcopy(q)
        p0_10,p1_01 = copy.deepcopy(p)
        
        # Compute the strain tensor ϵ = (Dq+Dq^T)/2
        ϵ00_00 = dl(q0_10,0) 
        ϵ11_00 = dl(q1_01,1) 
        ϵ01_11 = dr(q0_10,1) + dr(q1_01,0) # We omit the factor two in ϵ01 in view of Voigt's notation

        # Compute the stress tensor
        σ00_00 = C00_00*ϵ00_00+C01_00*ϵ11_00 
        σ11_00 = C10_00*ϵ00_00+C11_00*ϵ11_00 
        σ01_11 = C22_11*ϵ01_11 

        # Stress divergence
        dp0_10 = dr(σ00_00,0) + dl(σ01_11,1) 
        dp1_01 = dl(σ01_11,0) + dr(σ11_00,1)
#        self.tmp = (dp0_10,dp1_01),(σ00_00,σ11_00,σ01_11),(ϵ00_00,ϵ11_00,ϵ01_11)

        # Symplectic updates : first p, then q
        p0_10 += dt*dp0_10 
        p1_01 += dt*dp1_01 

        q0_10 += dt*p0_10*iρ_10 
        q1_01 += dt*p1_01*iρ_01

        return (q0_10,q1_01),(p0_10,p1_01)

class Virieux3:
    """The three dimensional Virieux scheme."""
    def __init__(self,ρ,C,stag):
        """
        Inputs : 
        - ρ : density, everywhere positive. 
        - C : Hooke tensor, assumes a VTI structure. Assumes C02=C12=0 and C symmetric.
        - stag : staggered grid difference operators
        """
        self.stag = stag
        ar = stag.avg_right
        # Variable locations on the grid shown after subscript
        iρ = 1/ρ
        self.iρ_100 = ar(iρ,0)
        self.iρ_010 = ar(iρ,1) 
        self.iρ_001 = ar(iρ,2) 
        self.C00_000 = C[0,0]
        self.C01_000 = C[0,1]
        self.C02_000 = C[0,2]
        self.C11_000 = C[1,1] 
        self.C12_000 = C[1,2] 
        self.C22_000 = C[2,2] 
        # Voigt : 3 -> (1,2), 4->(0,2), 5->(0,1)
        self.C33_011 = ar(C[3,3],axis=(1,2))
        self.C44_101 = ar(C[4,4],axis=(0,2))
        self.C55_110 = ar(C[5,5],axis=(0,1))

    def step(self,q,p,dt):
        """
        Inputs : 
        - q,p : position and momentum.
        - dt : time step.
        """
        # Variable locations shown after subscript
        dl,dr = self.stag.diff_left, self.stag.diff_right
        iρ_100,iρ_010,iρ_001, C00_000,C01_000,C02_000,C11_000,C12_000,C22_000, C33_011,C44_101,C55_110 = \
        self.iρ_100,self.iρ_010,self.iρ_001, self.C00_000,self.C01_000,self.C02_000,self.C11_000,self.C12_000,self.C22_000, self.C33_011,self.C44_101,self.C55_110

        q0_100,q1_010,q2_001 = copy.deepcopy(q)
        p0_100,p1_010,p2_001 = copy.deepcopy(p)
        
        # Compute the strain tensor ϵ = (Dq+Dq^T)/2
        ϵ00_000 = dl(q0_100,0) 
        ϵ11_000 = dl(q1_010,1)
        ϵ22_000 = dl(q2_001,2)
        ϵ01_110 = dr(q0_100,1) + dr(q1_010,0) # We omit the factor two in ϵ01 in view of Voigt's notation
        ϵ02_101 = dr(q0_100,2) + dr(q2_001,0) 
        ϵ12_011 = dr(q1_010,2) + dr(q2_001,1) 

        # Compute the stress tensor
        σ00_000 = C00_000*ϵ00_000 + C01_000*ϵ11_000 + C02_000*ϵ22_000
        σ11_000 = C01_000*ϵ00_000 + C11_000*ϵ11_000 + C12_000*ϵ22_000
        σ22_000 = C02_000*ϵ00_000 + C12_000*ϵ11_000 + C22_000*ϵ22_000
        # Voigt : 3 -> (1,2), 4->(0,2), 5->(0,1)
        σ12_011 = C33_011*ϵ12_011
        σ02_101 = C44_101*ϵ02_101
        σ01_110 = C55_110*ϵ01_110

        # Stress divergence
        dp0_100 = dr(σ00_000,0) + dl(σ01_110,1) + dl(σ02_101,2) 
        dp1_010 = dl(σ01_110,0) + dr(σ11_000,1) + dl(σ12_011,2) 
        dp2_001 = dl(σ02_101,0) + dl(σ12_011,1) + dr(σ22_000,2) 
        self.tmp = (dp0_100,dp1_010,dp2_001),(σ00_000,σ11_000,σ22_000,σ01_110,σ02_101,σ12_011),(ϵ00_000,ϵ11_000,ϵ22_000,ϵ01_110,ϵ02_101,ϵ12_011)

        # Symplectic updates : first p, then q
        p0_100 += dt*dp0_100 
        p1_010 += dt*dp1_010 
        p2_001 += dt*dp2_001 

        q0_100 += dt*p0_100*iρ_100 
        q1_010 += dt*p1_010*iρ_010
        q2_001 += dt*p2_001*iρ_001

        return (q0_100,q1_010,q2_001),(p0_100,p1_010,p2_001)

class Virieux1:
    """The two dimensional Virieux scheme."""
    def __init__(self,ρ,C,stag):
        """
        Inputs : 
        - ρ : density, everywhere positive. 
        - C : Hooke tensor, assumes a VTI structure. Assumes C02=C12=0 and C symmetric.
        - stag : staggered grid difference operators
        """
        self.stag = stag
        ar = stag.avg_right
        # Variable locations on the grid shown after subscript
        iρ = 1/ρ
        self.iρ_1 = ar(iρ,0)
        self.C00_0 = C[0,0]

    def step(self,q,p,dt):
        """
        Inputs : 
        - q,p : position and momentum.
        - dt : time step.
        """
        # Variable locations shown after subscript
        dl,dr = self.stag.diff_left, self.stag.diff_right
        C00_0,iρ_1 = self.C00_0,self.iρ_1

        q0_1, = copy.deepcopy(q)
        p0_1, = copy.deepcopy(p)
        
        # Compute the strain tensor ϵ = (Dq+Dq^T)/2
        ϵ00_0 = dl(q0_1,0) 

        # Compute the stress tensor
        σ00_0 = C00_0*ϵ00_0

        # Stress divergence
        dp0_1 = dr(σ00_0,0)

        # Symplectic updates : first p, then q
        p0_1 += dt*dp0_1 
        q0_1 += dt*p0_1*iρ_1 

        return (q0_1,),(p0_1,)

def VirieuxH1(ρ,C,stag,X):
    dl = stag.diff_left
    s = Virieux1(ρ,C,stag)
    def PotentialEnergy(q):
        q0_1, = q
        ϵ00_0 = dl(q0_1,0) # Compute the strain tensor ϵ = (Dq+Dq^T)/2
        return 0.5*s.C00_0*ϵ00_0**2
    def KineticEnergy(p):
        p0_1, = p
        return 0.5*s.iρ_1*p0_1**2
    H = QuadraticHamiltonian(PotentialEnergy,KineticEnergy)
    H.set_spmat(np.zeros_like(X)) # Replaces quadratic functions with sparse matrices
    return H

def VirieuxH2(ρ,C,stag,X,S=None):
    dl,dr = stag.diff_left,stag.diff_right
    s = Virieux2(ρ,C,stag)
    def PotentialEnergy(q):
        q0_10,q1_01 = q
        ϵ00_00 = dl(q0_10,0) # Compute the strain tensor ϵ = (Dq+Dq^T)/2
        ϵ11_00 = dl(q1_01,1) 
        ϵ01_11 = dr(q0_10,1) + dr(q1_01,0) # We omit the factor two in ϵ01 in view of Voigt's notation
        if S is not None: # Additional term arising in topographic changes of variables
            S00_00 = S[0,0]
            S11_00 = S[1,1]
            S01_11 = ar(2*S[0,1],(1+0,1+1))
            q_00 = ad.array((al(q0_10,0),al(q1_01,1)))
            q_11 = ad.array((ar(q0_10,1),ar(q1_01,1)))
            for ϵ_,S_,q_ in ((ϵ_00,S_00,q_00),(ϵ_11,S_11,q_11)): ϵ_ -= np.sum(S_*q_,axis=2)
        return 0.5*(s.C00_00*ϵ00_00**2 + 2*s.C01_00*ϵ00_00*ϵ11_00 + s.C11_00*ϵ11_00**2 + s.C22_11*ϵ01_11**2)
    def KineticEnergy(p):
        p0_10,p1_01 = p
        return 0.5*(s.iρ_10*p0_10**2 + s.iρ_01*p1_01**2)
    H = QuadraticHamiltonian(PotentialEnergy,KineticEnergy); H.set_spmat(np.zeros_like(X)); return H
    
def VirieuxH3(ρ,C,stag,X):
    dl,dr = stag.diff_left,stag.diff_right
    s = Virieux3(ρ,C,stag)
    def PotentialEnergy(q):
        q0_100,q1_010,q2_001 = q
        # Compute the strain tensor ϵ = (Dq+Dq^T)/2
        ϵ00_000 = dl(q0_100,0) 
        ϵ11_000 = dl(q1_010,1)
        ϵ22_000 = dl(q2_001,2)
        ϵ01_110 = dr(q0_100,1) + dr(q1_010,0) # We omit the factor two in ϵ01 in view of Voigt's notation
        ϵ02_101 = dr(q0_100,2) + dr(q2_001,0) 
        ϵ12_011 = dr(q1_010,2) + dr(q2_001,1) 
        # Compute the stress tensor
        σ00_000 = s.C00_000*ϵ00_000 + s.C01_000*ϵ11_000 + s.C02_000*ϵ22_000
        σ11_000 = s.C01_000*ϵ00_000 + s.C11_000*ϵ11_000 + s.C12_000*ϵ22_000
        σ22_000 = s.C02_000*ϵ00_000 + s.C12_000*ϵ11_000 + s.C22_000*ϵ22_000
        # Voigt : 3 -> (1,2), 4->(0,2), 5->(0,1)
        σ12_011 = s.C33_011*ϵ12_011
        σ02_101 = s.C44_101*ϵ02_101
        σ01_110 = s.C55_110*ϵ01_110
        return 0.5*(ϵ00_000*σ00_000 + ϵ11_000*σ11_000 + ϵ22_000*σ22_000 + ϵ01_110*σ01_110 + ϵ02_101*σ02_101 + ϵ12_011*σ12_011)
    def KineticEnergy(p):
        p0_100,p1_010,p2_001 = p
        return 0.5*(s.iρ_100*p0_100**2 + s.iρ_010*p1_010**2 + s.iρ_001*p2_001**2)
    H = QuadraticHamiltonian(PotentialEnergy,KineticEnergy); H.set_spmat(np.zeros_like(X)); return H

def eval_Lebedev(qfun,pfun,X,dx,t,dt):
    """
    Evaluate position and momentum at the given position and time, 
    taking into account spatial and temporal grid shifts
    """
    t2=t+dt/2
    vdim=len(X)
    if vdim==1:
        _,X_1 = shifted_grids(X,dx)
        return qfun(t2,X_1),pfun(t,X_1)
    if vdim==2:
        _,X_10,X_01,_ = shifted_grids(X,dx)
        q_10 = qfun(t2,X_10)
        q_01 = qfun(t2,X_01)
        p_10 = pfun(t ,X_10)
        p_01 = pfun(t ,X_01)
        return (q_10,q_01),(p_10,p_01) # Geometry is second
    if vdim==3:
        X_000,X_100,X_010,X_110,X_001,X_101,X_011,X_111 = shifted_grids(X,dx)
        q_100 = qfun(t2,X_100)
        q_010 = qfun(t2,X_010)
        q_001 = qfun(t2,X_001)
        q_111 = qfun(t2,X_111)
        p_100 = pfun(t ,X_100)
        p_010 = pfun(t ,X_010)
        p_001 = pfun(t ,X_001)
        p_111 = pfun(t ,X_111)
        return (q_100,q_010,q_001,q_111), (p_100,p_010,p_001,p_111)

class Lebedev2:
    """The two dimensional Lebedev scheme"""
    def __init__(self,ρ,C,stag):
        """
        Inputs : 
        - ρ : density, everywhere positive. 
        - C : Hooke tensor, assumed to be symmetric.
        - stag : staggered grid difference operators
        """
        self.stag = stag
        ar = stag.avg_right
        # Variable location on the grid shown after underscore
        iρ = 1/ρ 
        self.iρ_10 = ar(iρ,0) 
        self.iρ_01 = ar(iρ,1) 
        self.C_00 = C 
        self.C_11 = ar(C,axis=(2+0,2+1)) 

    def step(self,q,p,dt):
        """
        Inputs : 
        - q,p : position and momentum.
        - dt : time step.
        """
        dl,dr = self.stag.diff_left, self.stag.diff_right
        C_00,C_11,iρ_10,iρ_01 = self.C_00,self.C_11,self.iρ_10,self.iρ_01
         
        # Variable location on the grid shown after underscore
        (q0_10,q1_10), (q0_01,q1_01) = copy.deepcopy(q)
        (p0_10,p1_10), (p0_01,p1_01) = copy.deepcopy(p)

        # Compute the strain tensor ϵ = (Dq+Dq^T)/2
        ϵ00_00 = dl(q0_10,0) 
        ϵ11_00 = dl(q1_01,1) 
        ϵ01_00 = dl(q0_01,1)+dl(q1_10,0) 

        ϵ00_11 = dr(q0_01,0)
        ϵ11_11 = dr(q1_10,1)
        ϵ01_11 = dr(q0_10,1) + dr(q1_01,0) # We omit the factor two in ϵ01 in view of Voigt's notation
        
        # Compute the stress tensor
        σ00_00 = C_00[0,0]*ϵ00_00 + C_00[0,1]*ϵ11_00 + C_00[0,2]*ϵ01_00 
        σ11_00 = C_00[1,0]*ϵ00_00 + C_00[1,1]*ϵ11_00 + C_00[1,2]*ϵ01_00 
        σ01_00 = C_00[2,0]*ϵ00_00 + C_00[2,1]*ϵ11_00 + C_00[2,2]*ϵ01_00

        σ00_11 = C_11[0,0]*ϵ00_11 + C_11[0,1]*ϵ11_11 + C_11[0,2]*ϵ01_11 
        σ11_11 = C_11[1,0]*ϵ00_11 + C_11[1,1]*ϵ11_11 + C_11[1,2]*ϵ01_11 
        σ01_11 = C_11[2,0]*ϵ00_11 + C_11[2,1]*ϵ11_11 + C_11[2,2]*ϵ01_11

        # Stress divergence
        dp0_10 = dr(σ00_00,0) + dl(σ01_11,1) 
        dp1_01 = dl(σ01_11,0) + dr(σ11_00,1) 

        dp0_01 = dl(σ00_11,0) + dr(σ01_00,1) 
        dp1_10 = dr(σ01_00,0) + dl(σ11_11,1) 
#        self.tmp = (dp0_10,dp1_01),(σ00_00,σ11_00,σ01_11),(ϵ00_00,ϵ11_00,ϵ01_11)

        # Symplectic updates : first p, then q
        p0_10 += dt*dp0_10 
        p1_10 += dt*dp1_10 
        p0_01 += dt*dp0_01 
        p1_01 += dt*dp1_01 

        q0_10 += dt*p0_10*iρ_10
        q1_10 += dt*p1_10*iρ_10
        q0_01 += dt*p0_01*iρ_01
        q1_01 += dt*p1_01*iρ_01

        return ((q0_10,q1_10), (q0_01,q1_01)), ((p0_10,p1_10), (p0_01,p1_01))

class Lebedev3:
    """The three dimensional Lebedev scheme"""
    def __init__(self,ρ,C,stag):
        self.stag = stag
        ar = stag.avg_right
        # Variable location on the grid shown after underscore
        iρ = 1/ρ 
        self.iρ_100 = ar(iρ,0) 
        self.iρ_010 = ar(iρ,1)
        self.iρ_001 = ar(iρ,2)
        self.iρ_111 = ar(iρ,(0,1,2))
        self.C_000 = C 
        self.C_110 = ar(C,axis=(2+0,2+1)) 
        self.C_101 = ar(C,axis=(2+0,2+2)) 
        self.C_011 = ar(C,axis=(2+1,2+2)) 

    def step(self,q,p,dt):
        q_100,q_010,q_001,q_111 = copy.deepcopy(q) 
        p_100,p_010,p_001,p_111 = copy.deepcopy(p)
        dl,dr = self.stag.diff_left, self.stag.diff_right
        iρ_100,iρ_010,iρ_001,iρ_111, C_000,C_110,C_101,C_011 = self.iρ_100,self.iρ_010,self.iρ_001,self.iρ_111, self.C_000,self.C_110,self.C_101,self.C_011

        # Compute the strain tensor ϵ = (Dq+Dq^T)/2
        # Voigt convention : 00,11,22, 12,02,01
        ϵ_000 = (dl(q_100[0],0), dl(q_010[1],1), dl(q_001[2],2), dl(q_001[1],2)+dl(q_010[2],1), dl(q_001[0],2)+dl(q_100[2],0), dl(q_010[0],1)+dl(q_100[1],0) )
        ϵ_110 = (dr(q_010[0],0), dr(q_100[1],1), dl(q_111[2],2), dl(q_111[1],2)+dr(q_100[2],1), dl(q_111[0],2)+dr(q_010[2],0), dr(q_100[0],1)+dr(q_010[1],0) )
        ϵ_101 = (dr(q_001[0],0), dl(q_111[1],1), dr(q_100[2],2), dr(q_100[1],2)+dl(q_111[2],1), dr(q_100[0],2)+dr(q_001[2],0), dl(q_111[0],1)+dr(q_001[1],0) )
        ϵ_011 = (dl(q_111[0],0), dr(q_001[1],1), dr(q_010[2],2), dr(q_010[1],2)+dr(q_001[2],1), dr(q_010[0],2)+dl(q_111[2],0), dr(q_001[0],1)+dl(q_111[1],0) )

        # Compute the stress tensor, as a matrix-vector product
        σ_000 = [sum(C_000[i,j]*ϵ_000[j] for j in range(6)) for i in range(6)]
        σ_110 = [sum(C_110[i,j]*ϵ_110[j] for j in range(6)) for i in range(6)]
        σ_101 = [sum(C_101[i,j]*ϵ_101[j] for j in range(6)) for i in range(6)]
        σ_011 = [sum(C_011[i,j]*ϵ_011[j] for j in range(6)) for i in range(6)]

        # Stress divergence  [0,5,4]
        # Voigt convention   [5,1,3]
        #                    [4,3,2]
        dp_100 = (dr(σ_000[0],0) + dl(σ_110[5],1) + dl(σ_101[4],2), 
                  dr(σ_000[5],0) + dl(σ_110[1],1) + dl(σ_101[3],2), 
                  dr(σ_000[4],0) + dl(σ_110[3],1) + dl(σ_101[2],2) )
        
        dp_010 = (dl(σ_110[0],0) + dr(σ_000[5],1) + dl(σ_011[4],2), 
                  dl(σ_110[5],0) + dr(σ_000[1],1) + dl(σ_011[3],2), 
                  dl(σ_110[4],0) + dr(σ_000[3],1) + dl(σ_011[2],2) )
        
        dp_001 = (dl(σ_101[0],0) + dl(σ_011[5],1) + dr(σ_000[4],2), 
                  dl(σ_101[5],0) + dl(σ_011[1],1) + dr(σ_000[3],2), 
                  dl(σ_101[4],0) + dl(σ_011[3],1) + dr(σ_000[2],2) )
        
        dp_111 = (dr(σ_011[0],0) + dr(σ_101[5],1) + dr(σ_110[4],2), 
                  dr(σ_011[5],0) + dr(σ_101[1],1) + dr(σ_110[3],2), 
                  dr(σ_011[4],0) + dr(σ_101[3],1) + dr(σ_110[2],2) )

        self.tmp = (dp_100,dp_010,dp_001,dp_111), (σ_000,σ_110,σ_101,σ_011), (ϵ_000,ϵ_110,ϵ_101,ϵ_011)
        
        # Symplectic updates : first p, then q
        for i in range(3):
            p_100[i] += dt*dp_100[i]
            p_010[i] += dt*dp_010[i]
            p_001[i] += dt*dp_001[i]
            p_111[i] += dt*dp_111[i]

        for i in range(3):
            q_100[i] += dt*iρ_100*p_100[i]
            q_010[i] += dt*iρ_010*p_010[i]
            q_001[i] += dt*iρ_001*p_001[i]
            q_111[i] += dt*iρ_111*p_111[i]

        return (q_100,q_010,q_001,q_111), (p_100,p_010,p_001,p_111)

def LebedevH2(ρ,C,stag,X):
    dl,dr,al,ar = stag.diff_left,stag.diff_right,stag.avg_left,stag.avg_right
    s = Lebedev2(ρ,C,stag)
    def PotentialEnergy(q):
        q_10,q_01 = q
        # Compute the strain tensor ϵ = (Dq+Dq^T)/2. Voigt convention : 00,11, 01
        ϵ_00 = ad.array((dl(q_10[0],0),dl(q_01[1],1), dl(q_01[0],1)+dl(q_10[1],0)))
        ϵ_11 = ad.array((dr(q_01[0],0),dr(q_10[1],1), dr(q_10[0],1)+dr(q_01[1],0)))
        return 0.5*sum(lp.dot_VAV(ϵ_,C_,ϵ_) for (ϵ_,C_) in ((ϵ_00,s.C_00),(ϵ_11,s.C_11)))
    def KineticEnergy(p):
        p_10,p_01 = p
        return 0.5*sum(p_**2 * iρ_ for (p_,iρ_) in ((p_10,s.iρ_10),(p_01,s.iρ_01)))
    H = QuadraticHamiltonian(PotentialEnergy,KineticEnergy); H.set_spmat(np.zeros_like(X,shape=(2,*X.shape))); return H
     
def LebedevH3(ρ,C,stag,X):
    dl,dr = stag.diff_left,stag.diff_right
    s = Lebedev3(ρ,C,stag)
    def PotentialEnergy(q):
        q_100,q_010,q_001,q_111 = q
        # Compute the strain tensor ϵ = (Dq+Dq^T)/2. Voigt convention : 00,11,22, 12,02,01
        ϵ_000 = ad.array((dl(q_100[0],0), dl(q_010[1],1), dl(q_001[2],2), dl(q_001[1],2)+dl(q_010[2],1), dl(q_001[0],2)+dl(q_100[2],0), dl(q_010[0],1)+dl(q_100[1],0) ))
        ϵ_110 = ad.array((dr(q_010[0],0), dr(q_100[1],1), dl(q_111[2],2), dl(q_111[1],2)+dr(q_100[2],1), dl(q_111[0],2)+dr(q_010[2],0), dr(q_100[0],1)+dr(q_010[1],0) ))
        ϵ_101 = ad.array((dr(q_001[0],0), dl(q_111[1],1), dr(q_100[2],2), dr(q_100[1],2)+dl(q_111[2],1), dr(q_100[0],2)+dr(q_001[2],0), dl(q_111[0],1)+dr(q_001[1],0) ))
        ϵ_011 = ad.array((dl(q_111[0],0), dr(q_001[1],1), dr(q_010[2],2), dr(q_010[1],2)+dr(q_001[2],1), dr(q_010[0],2)+dl(q_111[2],0), dr(q_001[0],1)+dl(q_111[1],0) ))
        return 0.5*sum(lp.dot_VAV(ϵ_,C_,ϵ_) for (ϵ_,C_) in ((ϵ_000,s.C_000),(ϵ_110,s.C_110),(ϵ_101,s.C_101),(ϵ_011,s.C_011)))
    def KineticEnergy(p):
        p_100,p_010,p_001,p_111 = p
        return 0.5*sum(p_**2 * iρ_ for (p_,iρ_) in ((p_100,s.iρ_100),(p_010,s.iρ_010),(p_001,s.iρ_001),(p_111,s.iρ_111)))
    H = QuadraticHamiltonian(PotentialEnergy,KineticEnergy); H.set_spmat(np.zeros_like(X,shape=(4,*X.shape))); return H
    

def LebedevH2_ext(M,C,stag,X,S=None):
    dl,dr,al,ar = stag.diff_left,stag.diff_right,stag.avg_left,stag.avg_right
    M_10 = ar(M,2+0) 
    M_01 = ar(M,2+1) 
    C_00 = C 
    C_11 = ar(C,axis=(2+0,2+1)) 
    def PotentialEnergy(q):
        q_10,q_01 = q
        # Compute the strain tensor ϵ = (Dq+Dq^T)/2. Voigt convention : 00,11, 01
        ϵ_00 = ad.array((dl(q_10[0],0),dl(q_01[1],1), dl(q_01[0],1)+dl(q_10[1],0)))
        ϵ_11 = ad.array((dr(q_01[0],0),dr(q_10[1],1), dr(q_10[0],1)+dr(q_01[1],0)))
        if S is not None: # Additional term arising in topographic changes of variables
            S_00 = ad.array((S[0,0],S[1,1],2*S[0,1]))
            S_11 = ar(S_00,(2+0,2+1))
            q_00 = 0.5*(al(q_10,1+0)+al(q_01,1+1))
            q_11 = 0.5*(ar(q_10,1+1)+ar(q_01,1+0))
            for ϵ_,S_,q_ in ((ϵ_00,S_00,q_00),(ϵ_11,S_11,q_11)): ϵ_ -= np.sum(S_*q_,axis=1)
        return 0.5*sum(lp.dot_VAV(ϵ_,C_,ϵ_) for (ϵ_,C_) in ((ϵ_00,C_00),(ϵ_11,C_11)))
    def KineticEnergy(p):
        p_10,p_01 = p
        return 0.5*(p_10[None,:]*p_10[:,None]*M_10 + p_01[None,:]*p_01[:,None]*M_01) 
    H = QuadraticHamiltonian(PotentialEnergy,KineticEnergy); H.set_spmat(np.zeros_like(X,shape=(2,*X.shape))); return H

def _dispersion_DiffStaggered(s,h,order=2):
    """
    Approximation of identity, corresponding to the Fourier transform 
    of first order finite difference operators on staggered grids. Denoted I_order^h in text.
    """
    h2 = h/2; sh2 = s*h/2
    if order==2: return np.sin(sh2)/h2
    if order==4: return (9/8)*np.sin(sh2)/h2-(1/24)*np.sin(3*sh2)/h2
    raise ValueError("Unsupported order")

def _dispersion_AvgStaggered(s,h,order=2):
    """
    Fourier transform of the midpoint interpolation operator on staggered grid.
    """
    sh2 = s*h/2
    if order==2: return np.cos(sh2)
    if order==4: return (9/8)*np.cos(sh2)-(1/8)*np.cos(3*sh2)
    raise ValueError("Unsupported order")    

def _dispersion_Iinv(s,h):
    """Inverse of I(s,h,order=2), where I is _dispersion_DiffStaggered"""
    h2=h/2
    return np.arcsin(s*h2)/h2

def _dispersion_ElasticT2(ρ,Ck,dt):
    """Dispersion of a second order time discretization of an elastic wave equation"""
    if np.ndim(Ck)==1: Iω2,vq = np.linalg.eigh(Ck/ρ)
    else: Iω2,vq = np.linalg.eigh(np.moveaxis(Ck/ρ,(0,1),(-2,-1))); Iω2 = np.moveaxis(Iω2,-1,0); vq = np.moveaxis(vq,(-2,-1),(0,1)) 
    
    Iω = np.sqrt(Iω2)
    ω = _dispersion_Iinv(Iω,dt)
    vp = -ρ*Iω*vq # Omitting multiplication by i
    return ω,vq,vp

def dispersion_Virieux(k,ρ,C,dx,dt,order_x=2):
    """
    Dispersion relation for the Virieux and Lebedev schemes (but Virieux expects C02=C12=0)
    returns the wavemodes and pulsations corresponding to the wavenumber k.
    """
    Ik = _dispersion_DiffStaggered(k,dx,order_x)
    Ck = Hooke(C).contract(Ik)
    return _dispersion_ElasticT2(ρ,Ck,dt)

def mk_planewave_e(k,ω,vq,vp):
    """Make an elastic planewave with the specified parameters."""
    def brdcst(v,x): return fd.as_field(v,x[0].shape,conditional=False)
    def expi(t,x): return np.exp(1j*(lp.dot_VV(brdcst(k,x),x) - ω*t)) 
    def q_exact(t,x): return    expi(t,x) * brdcst(vq,x)
    def p_exact(t,x): return 1j*expi(t,x) * brdcst(vp,x)
    return q_exact,p_exact

def SellingCorrelated2H(ρ,C,X,dx,order_x=2,bc='Periodic'):
    λ,E = C if isinstance(C,tuple) else Hooke(C).Selling() # Note that E is a (collection of) symmetric matrices
    corr = lp.dot_VV(E[0],E[1]) # Correlation between the two offsets (do they point in the same direction)
    λ,corr = [fd.as_field(e,X[0].shape) for e in (λ,corr)]
    padding = AnisotropicWave.bc_to_padding[bc]
    def PotentialEnergy(q):
        dq0 = fd.DiffEll(q[0],E[0],dx,order=order_x,padding=padding)
        dq1 = fd.DiffEll(q[1],E[1],dx,order=order_x,padding=padding)
        sq_pos = np.sum((dq0+dq1      )**2,axis=0) # Best when positive correlation
        sq_neg = np.sum((dq0+dq1[::-1])**2,axis=0) # Best when negative correlation
        return 0.5 * λ * np.where(corr==0,(sq_pos+sq_neg)/2, # Symmetric version
                         np.where(corr>0,  sq_pos,sq_neg  )) # Asymmetric versions
    def KineticEnergy(p): return 0.5*p**2/ρ
    H = QuadraticHamiltonian(PotentialEnergy,KineticEnergy); H.set_spmat(np.zeros_like(X)); return H

def SellingCorrelated2H_ext(M,C,X,dx,order_x=2,S=None,bc='Periodic'):
    λ,E = C if isinstance(C,tuple) else Hooke(C).Selling() # Note that E is a (collection of) symmetric matrices
    corr = lp.dot_VV(E[0],E[1]) # Correlation between the two offsets (do they point in the same direction)
    λ,corr = [fd.as_field(e,X[0].shape) for e in (λ,corr)]
    padding = AnisotropicWave.bc_to_padding[bc]
    if S is None: ES = (0.,0.)
    else: ES = np.sum(E[:,:,None,:]*S[:,:,:,None],axis=(0,1))
    def PotentialEnergy(q):
        dq0 = fd.DiffEll(q[0],E[0],dx,order=order_x,α=ES[0]*q[0],padding=padding)
        dq1 = fd.DiffEll(q[1],E[1],dx,order=order_x,α=ES[1]*q[1],padding=padding)
        sq_pos = np.sum((dq0+dq1      )**2,axis=0) # Best when positive correlation
        sq_neg = np.sum((dq0+dq1[::-1])**2,axis=0) # Best when negative correlation
        return 0.5 * λ * np.where(corr==0,(sq_pos+sq_neg)/2, # Symmetric version
                         np.where(corr>0,  sq_pos,sq_neg  )) # Asymmetric versions
    def KineticEnergy(p): return 0.5*p[None,:]*p[:,None]*M
    H = QuadraticHamiltonian(PotentialEnergy,KineticEnergy); H.set_spmat(np.zeros_like(X)); return H

from agd.ExportedCode.Notebooks_Div.HighOrderWaves import _dispersion

def sin_eet(e,dx,order_x=2,sign=0):
    """Approximates of e e^T as dx->0, arises from FFT of scheme."""
    def ondiag(s): # Fourier transform of finite difference approximation of second derivative, see _γ
        if order_x==2: return 4*np.sin(s/2)**2
        if order_x==4: return (np.cos(2*s)-16*np.cos(s)+15)/6
        if order_x==6: return 49/18-3*np.cos(s)+3/10*np.cos(2*s)-1/45*np.cos(3*s)
    def offdiag_sym(s,t): # Fourier transform of symmetrized finite difference approximation of cross derivative
        if order_x==2: return np.sin(s)*np.sin(t)
        if order_x==4: return (8*np.sin(s)*np.sin(t) + (4*np.sin(s)-np.sin(2*s))*(4*np.sin(t)-np.sin(2*t)))/12
        if order_x==6: return (-9*np.sin(2*s)*(13*np.sin(t)-5*np.sin(2*t)+np.sin(3*t))+9*np.sin(s)*(50*np.sin(t)-13*np.sin(2*t)+2*np.sin(3*t))+np.sin(3*s)*(18*np.sin(t)-9*np.sin(2*t)+2*np.sin(3*t)))/180        
    def offdiag_asym(s,t): # Fourier transform of non-symmetrized finite difference approximation of cross derivative
        if order_x==2: return 1 - np.cos(s) + np.cos(s - t) - np.cos(t)
        if order_x==4: return -(1/12)*(-9+12*np.cos(s)-3*np.cos(2*s)+4*np.cos(s-2*t)-20*np.cos(s-t)-np.cos(2*(s-t))+4*np.cos(2*s-t)+12*np.cos(t)-3*np.cos(2*t)+4*np.cos(s+t))
        if order_x==6: return (1/180)*(101-153*np.cos(s)+63*np.cos(2*s)-11*np.cos(3*s)+18*np.cos(s-3*t)-9*np.cos(2*s-3*t)-108*np.cos(s-2*t)-9*np.cos(3*s-2*t)+342*np.cos(s-t)+45*np.cos(2*(s-t))+2*np.cos(3*(s-t))-108*np.cos(2*s-t)+18*np.cos(3*s-t)-153*np.cos(t)+63*np.cos(2*t)-11*np.cos(3*t)-108*np.cos(s+t)+9*np.cos(2*s+t)+9*np.cos(s+2*t))
    def offdiag(s,t,σ): # Fourier transform of finite difference approximation of cross derivative
        return offdiag_sym(s,t)-σ*(offdiag_sym(s,t)-offdiag_asym(s,t))
    vdim = len(e)
    return np.array([[ondiag(e[i]*dx) if i==j else offdiag(e[i]*dx,e[j]*dx,sign if vdim==2 else sign[3-i-j])
             for i in range(vdim)] for j in range(vdim)])/dx**2

def corr_signs(σ):
    """
    Input : a set of vdim = 2 or 3 vectors.
    Ouput : sums of best correlations σi σj, 0 <= i < j < vdim.
    """
    if len(σ)==2: return np.sign(lp.dot_VV(σ[:,0],σ[:,1])) # Best sign correlation in dimension 2
    # See discussion below about best sign correlations in dimension 3
    σs = np.array([lp.dot_VV(σ[:,1],σ[:,2]),lp.dot_VV(σ[:,0],σ[:,2]),lp.dot_VV(σ[:,0],σ[:,1])])
    ϵ = ((1,1,1),(-1,1,1),(1,-1,1),(-1,-1,1))
    ϵs = np.array([(ϵ1*ϵ2,ϵ0*ϵ2,ϵ0*ϵ1) for ϵ0,ϵ1,ϵ2 in ϵ])
    ϵs = np.expand_dims(ϵs,axis=tuple(range(-σ.ndim+2,0)))
    σϵs = np.sum(σs*ϵs,axis=1)
    σϵmax = σϵs==np.max(σϵs,axis=0)
    return np.sign(np.sum(ϵs*σϵmax[:,None],axis=0))

def sin_contract(C,k,dx,order_x):
    """Approximates Hooke(C).contract(k) as dx->0, arises from FFT of scheme."""
    λ,σ = C if isinstance(C,tuple) else Hooke(C).Selling()
    σk = lp.dot_AV(σ,k[:,None])
    σs = corr_signs(σ)
    return np.sum(λ*sin_eet(σk,dx,order_x,σs),axis=2)
    
def dispersion_SellingCorrelated(k,ρ,C,dx,dt,order_x=2):
    """Return all discrete propagation modes."""
    # For now, we assume that M = Id/ρ. Anisotropic M seem doable, but would require taking care
    # of the non-commutativity of a number of matrices in the scheme.
    from agd.ExportedCode.Notebooks_Div.HighOrderWaves import eig
    Ck = sin_contract(C,k,dx,order_x)
    return _dispersion_ElasticT2(ρ,Ck,dt)

def SellingCorrelatedH(ρ,C,X,dx,order_x=2,bc='Periodic'):
    λ,E = C if isinstance(C,tuple) else Hooke(C).Selling() # Note that E is a (collection of) symmetric matrices
    ϵ = corr_signs(E) # Correlation between the two offsets (do they point in the same direction)
    λ,ϵ = [fd.as_field(e,X[0].shape) for e in (λ,ϵ)]
    ϵ_pos = (ϵ>0)+(ϵ>=0).astype(int); ϵ_neg = (ϵ<0)+(ϵ<=0).astype(int) # Caution : array([True])+array([True]) == array([True])
    
    padding = AnisotropicWave.bc_to_padding[bc]
    def PotentialEnergy(q):
        dq = ad.array([fd.DiffEll(qi,Ei,dx,order=order_x,padding=padding) for qi,Ei in zip(q,E)])
        if vdim==2: return 0.5 * λ * (dq[0]**2+dq[1]**2 + ϵ_pos*dq[0]*dq[1] + ϵ_neg*dq[0]*dq[1,::-1])
        dq_pos = ad.array([dq[1]*dq[2],     dq[0]*dq[2],     dq[0]*dq[1]])
        dq_neg = ad.array([dq[1]*dq[2,::-1],dq[0]*dq[2,::-1],dq[0]*dq[1,::-1]])
        return 0.5 * λ * (dq**2 + ϵ_pos[:,None]*dq_pos + ϵ_neg[:,None]*dq_neg)
    def KineticEnergy(p): return 0.5*p**2/ρ
    H = QuadraticHamiltonian(PotentialEnergy,KineticEnergy); H.set_spmat(np.zeros_like(X)); return H

_precomp_offsets = np.array([[1,0,0,2,0,2],[0,1,0,0,2,2],[0,0,1,1,1,1]])
_precomp_inv = np.linalg.inv(flatten_symmetric_matrix(lp.outer_self(_precomp_offsets)))

def staggered_decomp_linprog(D,select='Cross'):
    """
    Modified Selling decomposition, for moderate anisotropy (all Thomsen materials execpt biotite crystal).
    - select ('Mean','Cross','Short'): selection principle for the decomposition
       'Mean' : average of all possible decompositions
       'Cross' : minimize the cross derivative coefficient, so as to achieve coef01**2 <= coef[0]*coef[1] when possible
       'Short' : minimize the coefficient of the longest offset, so as to improve stencil locality
    """
    # Single vertex, no need to find the best one in Ryskov's polyhedron.
    assert D.shape[:2]==(3,3)
    def af(x): return fd.as_field(x,D.shape[2:]) #np.expand_dims(x,tuple(range(2-D.ndim,0)))
    sdim = D.ndim-2 # number of additional dimensions
    D_flat = flatten_symmetric_matrix(D)
    signs = np.sign(D_flat[3:5])
    signs[signs==0]=1
    D_flat[3:5]*=signs
    if D_flat.ndim<=2: coefs = _precomp_inv@D_flat # Account for weird @ semantics
    else: coefs = np.moveaxis(_precomp_inv@np.moveaxis(D_flat,0,-2),-2,0)
    tmin = - np.min(coefs[[2,5]],axis=0)
    tmax =   np.min(coefs[[3,4]],axis=0)
    # We should have tmin<=tmax, up to machine precision, if the linear program is solvable. 
    # Possible selection principles for parameter t such that tmin<=t<=tmax : 
    if select=='Mean': t = (tmin+tmax)/2 # - midpoint, average of possible decompositions
    else: # Cross : minimize |coef01|. Short : minimize the last coefficient
        if select=='Cross': tmix = np.prod(signs,axis=0)*D[0,1]/4-coefs[-1]
        elif select=='Short': tmix = -coefs[-1]
        t = np.maximum(tmin,np.minimum(tmax,tmix))
    coefs += t*af(np.array([0,0,1,-1,-1,1.]))
    offsets = af(_precomp_offsets).copy()
    offsets[0:2]*=signs[:,None].astype(int)
    # Adjust for the coefficient at (O,1)
    coef01 = D[0,1] - np.sum(np.prod(offsets[0:2],axis=0)*coefs,axis=0)
    return (coefs[0],coefs[1],coef01),coefs[2:],offsets[:,2:]

def SellingStaggered2H(ρ,C,stag,X,S=None):
    """Hamiltonian of the 2d elastic wave equation, implemented using a staggered variant of Selling's decomposition."""
    dro,dl,ar = stag.diff_right_offset,stag.diff_left,stag.avg_right
    coefs_00,coefs_11,offsets_11 = staggered_decomp_linprog(C)
    def af(x): return fd.as_field(x,X[0].shape)
    iρ = af(1/ρ)    
    iρ_10 = ar(iρ,0)
    iρ_01 = ar(iρ,1)
    # TODO : higher spatial accuracy by properly locating coefs_11 (average with neighbors)
    assert S is None
    def PotentialEnergy(q):
        q0_10,q1_01 = q
        ϵ00_00 = dl(q0_10,0) # These terms handled similarly to Virieux
        ϵ11_00 = dl(q1_01,1) # Compute the strain tensor ϵ = (Dq+Dq^T)/2
        res = coefs_00[0]*ϵ00_00**2 + coefs_00[1]*ϵ11_00**2 +  2*coefs_00[2]*ϵ00_00*ϵ11_00
        for λ_11,offset_11 in zip(coefs_11,np.moveaxis(offsets_11,1,0)):
            e0,e1 = (offset_11[0],offset_11[2]),(offset_11[2],offset_11[1])
            ϵ_11 = dro(q0_10,1,e0) + dro(q1_01,0,e1)
            res += λ_11 * ϵ_11**2
        return 0.5*res
    def KineticEnergy(p):
        p0_10,p1_01 = p
        return 0.5*(iρ_10*p0_10**2 + iρ_01*p1_01**2)
    H = QuadraticHamiltonian(PotentialEnergy,KineticEnergy); H.set_spmat(np.zeros_like(X)); return H

def dispersion_SellingStaggered2(k,ρ,C,dx,dt,order_x=2,decomp=staggered_decomp_linprog):
    """
    Dispersion relation for the staggered Selling scheme for the 2d elastic wave equation.
    Returns. 
    - ω,vq,vp : pulsation, and amplitude of position and impulsion, of the wavemodes.
    """
    coefs_00,coefs_11,offsets_11 = decomp(C)
    def disp_stag(k): return _dispersion_DiffStaggered(k,dx,order_x)
    Ik = disp_stag(k)
    def af(x): return fd.as_field(x,k[0].shape)
    eye = af(np.eye(2))
    Ck_00 = coefs_00[0] * Ik[0]**2 * lp.outer_self(eye[0]) + coefs_00[1] * Ik[1]**2 * lp.outer_self(eye[1]) \
     + coefs_00[2]*Ik[0]*Ik[1]*(lp.outer(eye[0],eye[1])+lp.outer(eye[1],eye[0])) 
    Ck_11 = []
    for λ_11,offset_11 in zip(coefs_11,np.moveaxis(offsets_11,1,0)):
        e0,e1 = (offset_11[0],offset_11[2]),(offset_11[2],offset_11[1])
        e0,e1 = map(af,(e0,e1))
        Ik_11 = disp_stag(lp.dot_VV(k,e0)),disp_stag(lp.dot_VV(k,e1))
        Ck_11.append( λ_11 * lp.outer_self(Ik_11))
    return _dispersion_ElasticT2(ρ,Ck_00+np.sum(Ck_11,axis=0),dt)

_decomp_extended_precomp_vertices = np.array([[4.23205081,7.46410162,4.23205081,-3.73205081,-5.73205081,1],[1,4/3,3.30940108,-2/3,-7.60683603,2.97606774],[2.36602540,3.15470054,2.36602540,-1.57735027,-5.02072594,1.57735027],[1.5,1.15470054,1.5,-5.77350269e-01,-3,1],[2.17846097e+01,12,1,-2.88923048e+01,-6,9],[7.92820323,4,1,-5.96410162,-2,1],[1,1.92450090e-01,4/3,-9.62250449e-02,-2.66666667,1],[1,-1.28197512e-16,1,-0.5,-2,1],[2.42224319e+01,1.34074773e+01,1,-3.57411251e+01,-6.70373864,1.18149546e+01],[4.23205081,7.46410162,4.23205081,-5.73205081,-3.73205081,1],[4.78985145e+01,2.96201144e+01,2.46834287,-6.61769145e+01,-1.48100572e+01,2.07467429e+01],[7.46410162,1.49282032e+01,7.46410162,-8.21410162,-8.21410162,1],[6.36602540,3.09807621,1,-1.67320508e+01,-6,9],[4/3,1.92450090e-01,1,-2.66666667,-9.62250449e-02,1],[1,-1.28197512e-16,1,-2,-0.5,1],[1,4,7.92820323,-2,-5.96410162,1],[1.5,1.15470054,1.5,-3,-5.77350269e-01,1],[8.34298427,4.23947394,1,-2.16438643e+01,-6.47894789,1.09157915e+01],[3.30940108,4/3,1,-7.60683603,-2/3,2.97606774],[2.36602540,3.15470054,2.36602540,-5.02072594,-1.57735027,1.57735027],[1.07787504e+01,6.90264733,1.72566183,-2.84601481e+01,-1.03539710e+01,1.48052947e+01],[1,-1.28197512e-16,1,-2,-6,9],[1,-1.28197512e-16,1,-2,-2.5,2],[1.75621778e+01,-9.56217783,1,-16,-6,9],[6,2,1,-16,-6,9],[1,7.88675135e-01,2.36602540,-3.57735027,-6.30940108,4.15470054],[1.20291371,4.68609140e-01,1.20291371,-3.34304570,-3.34304570,2.87443656],[2.36602540,7.88675135e-01,1,-6.30940108,-3.57735027,4.15470054],[1,-1.28197512e-16,1,-2.5,-2,2],[4.46410162,2,1,-1.29282032e+01,-6,9],[1,1.34074773e+01,2.42224319e+01,-6.70373864,-3.57411251e+01,1.18149546e+01],[1,4.23947394,8.34298427,-6.47894789,-2.16438643e+01,1.09157915e+01],[1.72566183,6.90264733,1.07787504e+01,-1.03539710e+01,-2.84601481e+01,1.48052947e+01],[2.46834287,2.96201144e+01,4.78985145e+01,-1.48100572e+01,-6.61769145e+01,2.07467429e+01],[1,12,2.17846097e+01,-6,-2.88923048e+01,9],[1,3.09807621,6.36602540,-6,-1.67320508e+01,9],[1,2,4.46410162,-6,-1.29282032e+01,9],[1,-1.28197512e-16,1,-6,-2,9],[1,2,6,-6,-16,9],[1,-9.56217783,1.75621778e+01,-6,-16,9]])
_decomp_extended_precomp_offsets = np.array([[0,2,-2,0,0,2,-2,2,-2,1,-1,1,-1,4,-4,0,0,1,8.66025404e-01,0.5,6.12323400e-17,-0.5,-8.66025404e-01],[0,0,0,2,-2,2,2,-2,-2,1,1,-1,-1,0,0,4,-4,0,0.5,8.66025404e-01,1,8.66025404e-01,0.5],[1,1,1,1,1,1,1,1,1,2,2,2,2,1,1,1,1,0,0,0,0,0,0]])
_decomp_extended_precomp_active = np.array([[0,6,9,10,21,22],[3,6,9,10,17,22],[3,6,9,10,21,22],[0,3,9,10,21,22],[3,9,11,15,20,21],[0,3,9,11,20,21],[0,3,9,10,17,22],[0,3,9,17,19,21],[7,9,11,15,20,21],[0,7,9,11,21,22],[3,7,9,11,15,21],[0,6,7,9,21,22],[1,3,9,15,20,21],[0,1,9,11,20,21],[0,1,9,17,19,21],[0,1,9,10,17,22],[0,1,9,11,21,22],[1,7,9,15,20,21],[1,7,9,11,20,21],[1,7,9,11,21,22],[1,3,7,9,15,21],[3,5,15,17,19,21],[3,5,9,17,19,21],[3,5,9,15,19,20],[1,3,5,9,15,20],[1,3,5,9,17,22],[1,3,5,9,21,22],[1,3,5,9,20,21],[1,5,9,17,19,21],[1,3,5,15,20,21],[6,9,10,13,17,22],[3,6,9,13,17,22],[1,3,6,9,13,22],[1,6,9,10,13,22],[1,9,10,13,17,22],[1,3,9,13,17,22],[1,3,5,13,17,22],[1,5,13,17,19,21],[1,3,5,9,13,17],[1,5,9,13,17,18]])
_decomp_extended_precomp_inv = np.linalg.inv(np.moveaxis(flatten_symmetric_matrix(lp.outer_self(_decomp_extended_precomp_offsets[:,_decomp_extended_precomp_active])),0,1))
def staggered_decomp_linprogExt(D):
    shape = D.shape[2:]; assert D.shape[:2]==(3,3)
    D_flat = flatten_symmetric_matrix(D).reshape(6,-1) # Flatten additional shape
    # Find the best vertex (we are minimizing a linear form over a polyhedron)
    score = np.array([np.sum(D_flat* (np.array([1,s0*s1,1,s0,s1,1])*_decomp_extended_precomp_vertices)[:,:,None],axis=1) 
             for (s0,s1) in ((1,1),(1,-1),(-1,1),(-1,-1))])
    signs = np.argmin(np.min(score,axis=1),axis=0) 
    amin = np.argmin(np.take_along_axis(score,signs[None,None],axis=0)[0],axis=0)
    signs = np.array(((1,1),(1,-1),(-1,1),(-1,-1)))[signs].T
    ones = np.ones_like(signs[0])
    # Find the coefficients, by linear solve
    D_flat*=np.array([ones,signs[0]*signs[1],ones,signs[0],signs[1],ones])
    coefs = lp.dot_AV(np.moveaxis(_decomp_extended_precomp_inv[amin],0,-1),D_flat)
    offsets = np.moveaxis(_decomp_extended_precomp_offsets[:,_decomp_extended_precomp_active[amin]],-1,1)
    offsets[:2]*=signs[:,None]
    # Some post processing of the last offsets, so as to extract coef01, use integer coordinates...
    small = np.logical_and(offsets[2]==0,np.linalg.norm(offsets[:2],axis=0)<=1.1) # (cos θ, sin θ, 0) offsets receive special treatment
    D_small = np.sum(flatten_symmetric_matrix(lp.outer_self(offsets[:2])*np.where(small,coefs,0)),axis=1)
    coefs[small]=0; offsets[:,small]=0
    return D_small[[0,2,1]].reshape((3,*shape)),coefs[:-1].reshape((5,*shape)),offsets[:,:-1].astype(int).reshape((3,5,*shape))

def SellingStaggeredExt2H(ρ,C,stag,X,S=None):
    """Hamiltonian of the 2d elastic wave equation, implemented using a staggered variant of Selling's decomposition."""
    dlo,dro,dl,ar = stag.diff_left_offset,stag.diff_right_offset,stag.diff_left,stag.avg_right
    coefs_00,coefs,offsets = staggered_decomp_linprogExt(C)
    def af(x): return fd.as_field(x,X[0].shape)
    iρ = af(1/ρ)    
    iρ_10 = ar(iρ,0)
    iρ_01 = ar(iρ,1)
    assert S is None
    def PotentialEnergy(q):
        q0_10,q1_01 = q
        ϵ00_00 = dl(q0_10,0) # These terms handled similarly to Virieux
        ϵ11_00 = dl(q1_01,1) # Compute the strain tensor ϵ = (Dq+Dq^T)/2
        res = coefs_00[0]*ϵ00_00**2 + coefs_00[1]*ϵ11_00**2 +  2*coefs_00[2]*ϵ00_00*ϵ11_00
        for λ,offset in zip(coefs,np.moveaxis(offsets,-1,0)):
            if np.allclose(λ,0): continue
            e0,e1 = (offset[0],offset[2]),(offset[2],offset[1])
            at_11 = offset[2]%2==1
            ϵ = dlo(q0_10,0+at_11,e0,right=at_11) + dlo(q1_01,1-at_11,e1,right=at_11)
            # ϵ_00 = dlo(q0_10,0,e0) + dlo(q1_01,1,e1); ϵ_11 = dro(q0_10,1,e0) + dro(q1_01,0,e1); ϵ = np.where(at_11,ϵ_11,ϵ_00)
            # TODO : higher spatial accuracy by properly locating λ when at_11 is true (average with neighbors)
            res += λ * ϵ**2
        return 0.5*res
    def KineticEnergy(p):
        p0_10,p1_01 = p
        return 0.5*(iρ_10*p0_10**2 + iρ_01*p1_01**2)
    H = QuadraticHamiltonian(PotentialEnergy,KineticEnergy); H.set_spmat(np.zeros_like(X)); return H

def _staggered_decomp_linprogExt(m,offsets=_decomp_extended_precomp_offsets,offset_cost=None):
    # Setup and solve the linear program
    A_eq = flatten_symmetric_matrix(lp.outer_self(offsets))
    b_eq = flatten_symmetric_matrix(m)
    if offset_cost is None: c = -np.ones(A_eq.shape[1]) # Could try to adjust this objective to minimize distortion
    else: c = - np.array(list(map(offset_cost,offsets.T)))
    bounds = (0,None)
    res = linprog(c,A_eq=A_eq,b_eq=b_eq,bounds=bounds)
    #print(c); print(res.fun)
    coefs = res.x
    small = np.logical_and(offsets[2]==0,np.linalg.norm(offsets[:2],axis=0)<=1.1) # (cos θ,sin θ,0) offsets receive special treatment
    D_small = np.sum(flatten_symmetric_matrix(lp.outer_self(offsets[:2])*np.where(small,coefs,0)),axis=1)
    coefs[small]=0; offsets = offsets.copy(); offsets[:,small]=0
    select = coefs>0
    return D_small[[0,2,1]],coefs[select],offsets[:,select].astype(int)

def AcousticCenteredH(ρ,D,X,dx,order_x=2,bc='Periodic'):
    padding = {'Periodic':None,'Dirichlet':0}[bc]
    vdim = len(D)
    def PotentialEnergy(q): # q is stored at grid points
        e = np.eye(vdim).astype(int)
        dq = fd.DiffCentered(q,e,dx,order_x,padding=padding) # Centered finite differences, periodic b.c
        dq2 = np.sum(fd.DiffEll(q,e,dx,order_x,padding=padding)**2,axis=0) # Upwind and downwind finite differences, squared and summed.
        diag = sum(D[i,i]*dq2[i] for i in range(vdim)) # Sum of squares for the diagonal
        offdiag = sum(2*D[i,j]*dq[i]*dq[j] for i in range(vdim) for j in range(i)) if vdim>=2 else 0
        return 0.5*(diag+offdiag)
    def KineticEnergy(p):
        return 0.5*p**2/ρ
    H = QuadraticHamiltonian(PotentialEnergy,KineticEnergy); H.set_spmat(np.zeros_like(X[0])); return H

def AcousticCrissCrossH(ρ,D,X,dx,stag):
    dl,dr,al,ar = stag.diff_left,stag.diff_right,stag.avg_left,stag.avg_right
    vdim = len(D)
    iρ_1d = ar(1/ρ,tuple(range(vdim))) # 1d means At position 1, 11, or 111, in dimension d = 1, 2 or 3
    def PotentialEnergy(q_1d): 
        # Compute a centered gradient using centered averages and differences
        dq_0d = ad.array([al(dl(q_1d,i),tuple(range(i))+tuple(range(i+1,vdim))) for i in range(vdim)])
        return 0.5 * lp.dot_VAV(dq_0d,D,dq_0d)
    def KineticEnergy(p_1d):
        return 0.5 * p_1d**2 * iρ_1d
    H = QuadraticHamiltonian(PotentialEnergy,KineticEnergy); H.set_spmat(np.zeros_like(X[0])); return H

def _dispersion_DiffCentered(s,h,order_x):
    """Fourier transforme of the centered finite difference approximation of first derivative."""
    sh = s*h
    if order_x==2: return np.sin(sh)/h
    if order_x==4: return ((4/3)*np.sin(sh)-(1/6)*np.sin(2*sh))/h
    if order_x==6: return ((3/2)*np.sin(sh)-(3/10)*np.sin(2*sh)+(1/30)*np.sin(3*sh))/h

def _dispersion_Diff2(s,h,order_x): 
    """Fourier transform of finite difference approximation of second derivative."""
    sh = s*h
    if order_x==2: return 4*np.sin(sh/2)**2/h**2
    if order_x==4: return (np.cos(2*sh)-16*np.cos(sh)+15)/(6*h**2)
    if order_x==6: return (49/18-3*np.cos(sh)+3/10*np.cos(2*sh)-1/45*np.cos(3*sh))/h**2

def _dispersion_AcousticT2(ρ,Iω2,dt):
    """Helper implementing the dispersion for a second order accurate discretization in time of a scalar equation"""
    Iω = np.sqrt(Iω2)
    ω = _dispersion_Iinv(Iω,dt)
    vq = 1. # Amplitude of position q
    vp = -Iω*ρ*vq # Amplitude of impulsion p
    return ω,vq,vp

def dispersion_AcousticCentered(k,ρ,D,dx,dt,order_x=2):
    """Dispersion relation for the centered non-monotone finite differences approximation of the acoustic wave equation."""
    dk  = _dispersion_DiffCentered(k,dx,order_x)
    dk2 = _dispersion_Diff2(k,dx,order_x)
    vdim=len(k)
    diag = sum(D[i,i]*dk2[i] for i in range(vdim)) 
    offdiag = sum(2*D[i,j]*dk[i]*dk[j] for i in range(vdim) for j in range(i))
    Iω2 = (diag+offdiag)/ρ
    return _dispersion_AcousticT2(ρ,Iω2,dt)

def dispersion_AcousticCrissCross(k,ρ,D,dx,dt,order_x=2):
    """Dispersion relation for the criss-cross non-monotone finite differences approximation of the acoustic wave equation."""
    vdim = len(k)
    dk = np.array([np.prod([_dispersion_DiffStaggered(k[j],dx,order_x) if i==j else _dispersion_AvgStaggered(k[j],dx,order_x)
                        for j in range(vdim)],axis=0) for i in range(vdim)])
    Iω2 = lp.dot_VAV(dk,D/ρ,dk)
    return _dispersion_AcousticT2(ρ,Iω2,dt)


# Code automatically exported from notebook RollingBall_Models.ipynb in directory Notebooks_Algo
# Do not modify
from ... import AutomaticDifferentiation as ad
from ... import LinearParallel as lp
from ... import Metrics
from ... import FiniteDifferences as fd
from ... import Sphere as sp
norm = ad.Optimization.norm

rotation_from_quaternion = sp.rotation3_from_sphere3
quaternion_from_rotation = sp.sphere3_from_rotation3

quaternion_from_euclidean = sp.sphere_from_plane
euclidean_from_quaternion = sp.plane_from_sphere

euclidean_from_rotation = sp.ball3_from_rotation3
rotation_from_euclidean = sp.rotation3_from_ball3

def antisym(a,b,c):
    z=np.zeros_like(a)
    return ad.array([[z, -c, b], [c, z, -a], [-b, a, z]])

def exp_antisym(a,b,c):
    """Matrix exponential of antisym(a,b,c).
    Note : (a,b,c) is the axis of rotation."""
    s = ad.asarray(a**2+b**2+c**2)
    s[s==0]=1e-20 # Same trick as in numpy's sinc function ...
    sq = np.sqrt(s)
    co,si = np.cos(sq),np.sin(sq)
    cosc,sinc = (1-co)/s,si/sq    
    return ad.array([
        [co+cosc*a**2, cosc*a*b-sinc*c, cosc*a*c+sinc*b],
        [cosc*a*b+sinc*c, co+cosc*b**2, cosc*b*c-sinc*a],
        [cosc*a*c-sinc*b, cosc*b*c+sinc*a, co+cosc*c**2]])

def advance(state,control):
    """Move from a state to another by applying a control during a unit time"""
    state,control = map(ad.asarray,(state,control))
    state_physical = state[:-3]
    state_physical = state_physical + 0.25*control[:len(state_physical)] # Additive action on the physical state
    
    state_angular,qRef = rotation_from_euclidean(state[-3:])
    state_angular = lp.dot_AA(state_angular,exp_antisym(*control)) # Left invariant action
    
    return np.concatenate([state_physical,euclidean_from_rotation(state_angular,qRef)],axis=0)

def make_hamiltonian(controls,advance=advance):
    """Produces the hamiltonian function associated to a sub-Riemannian model, 
    defined by its controls and the advance function"""
    def hamiltonian(state):
        """The hamiltonian, a quadratic form on the co-tangent space"""
        # Array formatting to apply to several states simultanously
        state=ad.asarray(state); controls_ = fd.as_field(controls,state.shape[1:],depth=1) 
        
        grad = advance(state,controls_).gradient()
        return lp.dot_AA(lp.transpose(grad),grad)
    return hamiltonian


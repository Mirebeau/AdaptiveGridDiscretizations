import numpy as np
from .base import Base
from .riemann import Riemann
from .isotropic import Isotropic
from .. import AutomaticDifferentiation as ad
from .. import LinearParallel as lp
from ..FiniteDifferences import common_field


class AsymIso(Base):
	r"""
	A special case of AsymQuad metric taking the form
	$
	F(x) = \sqrt{a^2 |x|^2+sign(a) max(0,<w,x>)^2}
	$
	where $a$ is a field of scalars, and w a vector field.
	Member fields and __init__ arguments : 
	- a : an array of shape (n1,..,nk)
	- w : an array of shape (vdim,n1,...,nk)
	"""
	def __init__(self,a,w):
		a,w = (ad.asarray(e) for e in (a,w))
		self.a,self.w =common_field((a,w),(0,1))

	def norm(self,v):
		v,a,w = common_field((ad.asarray(v),self.a,self.w),(1,0,1))
		return np.sqrt(a**2*np.sum(v**2,axis=0) + np.sign(a)*np.maximum(lp.dot_VV(w,v),0.)**2)

	def gradient(self,v):
		v,a,w = common_field((ad.asarray(v),self.a,self.w),(1,0,1))
		g = a**2*v + np.sign(a)*np.maximum(0.,lp.dot_VV(w,v))*w
		return g/np.sqrt(lp.dot_VV(v,g))

	def dual(self):
		a2,s,w2 = self.a**2,np.sign(self.a),self._norm2w()
		r2 = s*(1/a2-1/(a2+s*w2))/w2
		return AsymIso(-1/self.a,np.sqrt(r2)*self.w)

	@property
	def vdim(self): return len(self.w)

	@property
	def shape(self): return self.a.shape

	def _norm2w(self): return np.sum(self.w**2,axis=0)
	def is_definite(self):
		return np.where(self.a>0,1.,self.a**2-self._norm2w()) > 0
	def anisotropy(self):
		a2,w2 = self.a**2,self._norm2w()
		return np.sqrt(np.where(self.a>0,1.+w2/a2,1./(1.-w2/a2)))
	def cost_bound(self):
		return np.where(self.a>0,np.sqrt(self.a**2+self._norm2w()),-self.a)

	def rotate(self,r): return AsymIso(self.a,lp.dot_AV(r,self.w))
	def with_cost(self,cost): return AsymIso(cost*self.a,cost*self.w)
	def with_costs(self,costs):
		if len(costs)>1 and not np.allclose(costs[1:],costs[0]):
			raise ValueError("Costs must be identical along all axes for Metrics.Isotropic. Consider using Metrics.Diagonal class")
		return self.with_cost(costs[0])

	def model_HFM(self):
		return 'AsymIso'+str(self.vdim)

	@classmethod
	def from_cast(cls,metric):
		if isinstance(metric,cls): return metric
		iso = Isotropic.from_cast(metric)
		w = np.zeros_like(iso.cost,shape=(iso.vdim,)+iso.shape)
		return cls(iso.cost,w)

	def __iter__(self):
		yield self.a
		yield self.w

	def make_proj_dual(self,**kwargs):
		"""kwargs : passed to Riemann.make_proj_dual"""
		vdim,a = self.vdim,self.a
		proj_a = Isotropic(np.abs(a)).make_proj_dual()
		eye = np.eye(vdim,like=a).reshape( (vdim,vdim)+(1,)*a.ndim )
		proj_w=Riemann(a**2*eye+np.sign(a)*lp.outer_self(self.w)).make_proj_dual(**kwargs)

		def proj(x):
			x,w = common_field((x,self.w),depths=(1,1))
			x_a = proj_a(x)
			x_w = proj_w(x)
			s = lp.dot_VV(w,x_a) > 0
			return np.where(s[None],x_w,x_a)
		return proj



	

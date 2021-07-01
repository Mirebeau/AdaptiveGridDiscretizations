# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import scipy.sparse

from .. import AutomaticDifferentiation as ad
from .. import LinearParallel as lp
from .hamiltonian_base import HamiltonianBase,fixedpoint

class MetricHamiltonian(HamiltonianBase):
	r"""
	Hamiltonian defined by an interpolated metric, which is dualized and interpolated.
	$$
	H(q,p) = \frac 1 2 F^*_q(p)^2
	$$
	__init__ arguments :
	- metric : dual defines the hamiltonian
	- inv_inner : passed to HamiltonianBase
	- **kwargs : passed to metric.dual().set_interpolation 
	"""

	def __init__(self,metric,inv_inner=None,**kwargs):
		super(MetricHamiltonian,self).__init__(inv_inner,vdim=metric.vdim)
		self._dualmetric = metric.dual()
		self._dualmetric.set_interpolation(**kwargs)

	def _H(self,q,p): return self._dualmetric.at(q).norm2(p)
	
	def _DqH(self,q,p): 
		q_ad = ad.Dense.identity(constant=q,shape_free=self.shape_free)
		return self._dualmetric.at(q_ad).norm2(p).gradient()

	def _DpH(self,q,p): return self._dualmetric.at(q).gradient2(p)

class GenericHamiltonian(HamiltonianBase):
	r"""
	Hamiltonian defined by a arbitrary function $f$ of two variables, 
	the position $q$ and impulsion $p$.
	$$
	H(q,p) = f(q,p)
	$$

	__init__ arguments : 
	- func : the function $f$ defining the hamiltonian, must take two arguments.
	- disassociate_ad (optional) : hide AD information when calling $f$. (Use to avoid 
	conflicts if the definition of $f$ itself involves automatic differentiation.)
	- shape_free (optional) : shape of position and momentum variables
	- **kwargs : passed to HamiltonianBase
	"""

	def __init__(self,func, disassociate_ad=False, **kwargs):
		super(GenericHamiltonian,self).__init__(**kwargs)
		self._func = func
		self.disassociate_ad = disassociate_ad

	def _H(self,q,p): return self._func(q,p)

	def _identity_ad(self,x,noad=None):
		x_ad = ad.Dense.identity(constant=x,shape_free=self.shape_free) 
		if self.disassociate_ad: 
			x_dis = ad.disassociate(x_ad,shape_free=self.shape_free)
			if noad is None: return x_dis
			else:return (x_dis,ad.disassociate(type(x_ad)(noad),shape_free=self.shape_free)) 
		else: 
			return x_ad if noad is None else (x_ad,noad)

	def _gradient_ad(self,x):
		if self.disassociate_ad: x=ad.associate(x)
		return x.gradient()

	def _DqH(self,q,p):
		q_ad,p = self._identity_ad(q,noad=p)
		return self._gradient_ad(self._H(q_ad,p))

	def _DpH(self,q,p):
		p_ad,q = self._identity_ad(p,noad=q)
		return self._gradient_ad(self._H(q,p_ad))

class SeparableHamiltonian(HamiltonianBase):
	"""
	Separable Hamiltonian defined by a pair of functions.
	$$
		H(q,p) = H_Q(q) + H_P(p).
	$$
	__init__ arguments : 
	- Hq,Hp : the two functions $H_Q,H_P$, of a single argument, defining the hamiltonian
	- **kwargs : passed to HamiltonianBase
	"""

	def __init__(self,Hq,Hp,**kwargs):
		super(SeparableHamiltonian,self).__init__(**kwargs,is_separable=True)
		self.Hq = Hq
		self.Hp = Hp

	def _H(self,q,p): return self.Hq(q) + self.Hp(p)

	def _DqH(self,q,_):
		q_ad = ad.Dense.identity(constant=q,shape_free=self.shape_free)
		return self.Hq(q_ad).gradient()

	def _DpH(self,_,p):
		p_ad = ad.Dense.identity(constant=p,shape_free=self.shape_free)
		return self.Hp(p_ad).gradient()

class QuadraticHamiltonian(HamiltonianBase):
	r"""
	Quadratic Hamiltonian, defined by a pair of linear operators.
	(Expected to be symmetric semi-definite.)
	$$
		H(q,p) = \frac 1 2 (< q, M_Q q > +< p, M_P p >).
	$$

	__init__ arguments : 
	- Mq,Mp : positive semi-definite matrices $M_Q,M_P$, typically given in sparse form.
	 Alternatively, define Mq,Mp as functions, and use the set_spmat
	 to automatically generate the sparse matrices using automatic differentiation.
	"""

	def __init__(self,Mq,Mp,**kwargs):
		super(QuadraticHamiltonian,self).__init__(**kwargs,is_separable=True)
		self.Mq = Mq
		self.Mp = Mp
	
	def flat(self,x): 
		return x.reshape((self.size_free,*x.shape[self.ndim_free:])) 

	def _H(self,q,p):
		q,p = map(self.flat,(q,p))
		return 0.5*(lp.dot_VV(q,ad.apply_linear_mapping(self.Mq,q)) 
			+ lp.dot_VV(p,ad.apply_linear_mapping(self.Mp,p)) )  

	def _DqH(self,q,_): return ad.apply_linear_mapping(self.Mq,self.flat(q))
	def _DpH(self,_,p): return ad.apply_linear_mapping(self.Mp,self.flat(p))

	@classmethod
	def spmat(cls,f,x,simplify_ad=None):
		"""
		Returns the sparse matrix associated to the hessian of f at x,
		generated using automatic differentiation.
		Output of f is summed, if non-scalar.
		- simplify_ad (optional): wether to simplify the ad information 
		   before generating the sparse matrix
		- inv_inner : inverse inner product, for gradient normalization
		"""
		x_ad = ad.Sparse2.identity(constant=x)
		f_ad = f(x_ad) 
		if simplify_ad is None: simplify_ad = f_ad.ndim > 0
		if simplify_ad: f_ad.simplify_ad(atol=True,rtol=True)
		return f_ad.hessian_operator( shape=(x_ad.size,x_ad.size) )

	def set_spmat(self,x,**kwargs):
		"""
		Replaces Mq,Mp and inv_inner with suitable sparse matrices, generated by spmat,
		if they are callables.
		"""
		self.shape_free = x.shape
		if callable(self.Mq): self.Mq = self.spmat(self.Mq,x,**kwargs)
		if callable(self.Mp): self.Mp = self.spmat(self.Mp,x,**kwargs)
		if callable(self.inv_inner): self.inv_inner = self.spmat(self.inv_inner,x,**kwargs) 





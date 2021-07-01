# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0


import itertools
import copy
import numpy as np
from ..Interpolation import map_coordinates,origin_scale_shape #UniformGridInterpolation

def odeint_array(f,y,t,grid,t_delay=0,t_substeps=2,order=1,**kwargs):
	"""
	Solve an ODE where the vector field is interpolated from an array.
	The vector field is linearly interpolated, and the Euler midpoint scheme is used.

	Inputs : 
	- f : generator of the vector fields to be interpolated, len(next(f))=vdim
	- y : initial value of the solution, len(y) = vdim 
	- t : time steps
	- grid : interpolation grid for the vector field f
	- t_delay : number of time steps to drop initially 
	- t_substeps : number of substeps
	- order : passed to odeint_array
	- **kwargs : passed to UniformGridInterpolation
	"""
	nt = len(t)
	solution = np.full_like(y,np.nan,shape=(nt,*y.shape))
	y0 = copy.copy(y)
	fit = iter(f)
	origin,scale,_ = origin_scale_shape(grid)
	def I(values,positions): # Interpolation
		return map_coordinates(values,positions,origin=origin,scale=scale,
			depth=1,order=order,**kwargs)

	f1 = next(fit)
	for i in range(nt-1):
		f0,f1 = f1,next(fit) # Vector fields to be interpolated
		t0,t1 = t[i],t[i+1]
		dt = (t1-t0)/t_substeps
		solution[i][:,t_delay==i] = y[:,t_delay==i] # Insert initial values
		valid = t_delay <= i 
		y0 = solution[i] if np.all(valid) else solution[i][:,valid]
		for s in range(t_substeps):
			w = s/t_substeps
			y1 = y0 + 0.5*dt * ((1-w)*I(f0,y0) + w*I(f1,y0)) # Compute midpoint
			w = (s+0.5)/t_substeps
			y0 = y0 +     dt * ((1-w)*I(f0,y1) + w*I(f1,y1)) # Make step
		if np.all(valid): solution[i+1] = y0
		else: solution[i+1][:,valid]=y0 

	return solution

class RecurseRewind:
	r"""
	This class is designed to iterate a function, and then roll back the iterations, 
	with a limited memory usage. For that purpose appropriate keypoints are stored,
	 in a multilevel manner.
	__init__ args: 
	 - next : method to compute the next step.
	 - initial : initial value
	 - params : passed to next, in addition to current value
	 - base : base $b$ used to internally represent the iteration counter, which balances 
	 a tradeoff between memory usage and computational cost. Iterating $O(b^n)$ times
	  and then rewinding these iterations, has a storage cost of $O(n*b)$ and a 
	  computational cost of $O(n * b^n)$ function evaluations.

	members:
	 - reversed : wether __next__ should advance or rewind the iterations
	"""

	def __init__(self,next,initial,params=tuple(),base=10):
		self.next = next
		self.params = params
		self._base = base
		self._keys = {0:initial}
		self._index = 0
		self.reversed = False
		self._initial_value = True

	def __iter__(self):
		self._initial_value = True
		return self

	def __next__(self):
		if self._initial_value: self._initial_value = False
		elif self.reversed: self.rewind()
		else: self.advance()
		return self.value()

#	def __reversed__(self): # No way to make this consistent and usable
#		self.reversed = True
#		return self

	def advance(self):
		self._keys[self.index+1] = self.next(self._keys[self.index],*self.params)
		self._index += 1
		# Delete previous keys to save storage
		for i in range(self._basepow):
			k = self.base**i
			for j in range(1,self.base): 
				self._keys.pop(self.index - j*k)

	def rewind(self):
		if self.index<=0: raise StopIteration
		self._keys.pop(self.index)
		i = self._basepow
		if i:
			k = self.base**i
			self._index -= k
			for _ in range(k-1):
				self.advance()
		else:
			self._index -= 1

	@property
	def _basepow(self):
		"""Returns the largest i such that self.base**i divides self.index"""
		assert self.index!=0
		basepow = self.base
		i = 0
		while self.index%basepow == 0:
			basepow *= self.base
			i+=1
		return i

	@property
	def index(self):
		"""The index of the current iteration"""
		return self._index
	def value(self):
		"""The value of the current iteration"""
		return self._keys[self.index]
	@property
	def base(self):return self._base

	
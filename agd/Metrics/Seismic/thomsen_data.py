# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

"""
We reproduce in this file list of VTI examples taken from "Weak elastic anisotropy" (Thomsen, 1986), 
Units are : 
- Vp,Vs : m/s
- 'ε','η','δ','γ' : dimensionless. (Originally, η is listed as δ^star.)
- ρ : g/cm^3


We also provide conversion utilities
"""


import numpy as np
from collections import OrderedDict,namedtuple

# ------ Thomsen Elastica materials -----

# At the time of writing, namedtuple requires normalized unicode characters.
# This is why the 'ε' is used instead of 'ϵ'.
# In jupyter notebooks : ε = \varepsilon, and ϵ = \epsilon. On the Greek keyboard, e -> ε.
# import unicodedata; 'ε' == unicodedata.normalize('NFKC','ϵ')
# https://stackoverflow.com/a/30505623/12508258


HexagonalMaterial = namedtuple('HexagonalMaterial',['c11','c12','c13','c33','c44'])

def get_δ(Vp,Vs,ϵ,η):
	"""
	Reconstructs Thomsen parameter δ, based on the other parameters, 
	and the formula in the Thomsen paper.
	(Note : some of the published ThomsenData appears to be inconsistent in this regard.)
	"""
	return 0.5*(ϵ + η/(1.-Vs**2/Vp**2))

TEM_data = namedtuple('ThomsenElasticMaterial_data',['Vp','Vs','ε','η','δ','γ','ρ'])
class ThomsenElasticMaterial(TEM_data):
	@classmethod
	def units(cls): return cls('m/s','m/s',1,1,1,1,'g/cm^3')

	@classmethod
	def from_hexagonal(cls,hex,ρ):
		"""
		Produces the Thomsen parameters, from the coefficients 'hex' of 
		a Hooke tensor with hexagonal symmetry, and the density 'ρ'.
		"""
		c11,c12,c13,c33,c44 = hex
		c66 = (c11-c12)/2

		Vp = np.sqrt(c33)
		Vs = np.sqrt(c44)

		ε = (c11-c33)/(2*c33)
		γ = (c66-c44)/(2*c44)
		η = (2*(c13+c44)**2 - (c33-c44)*(c11+c33-2*c44))/(2*c33**2)
		δ = get_δ(Vp,Vs,ϵ,η)

		return cls(Vp,Vs,ε,η,δ,γ,ρ)

	def to_hexagonal(self):
		"""
		Returns the coefficients of the reduced Hooke tensor, with hexagonal symmetry, and the density.
		Units. Reduced hooke tensor : (m/s)^2, density : g/cm^3
		"""
		Vp,Vs,ε,η,δ,γ,ρ = self
		c33 = Vp**2
		c44 = Vs**2

		c11 = 2*c33*ε+c33
		c66 = 2*c44*γ+c44
		c12 = c11-2*c66
		x = η*c33**2 + 0.5*(c33-c44)*(c11+c33-c44)
		c13 = np.sqrt(x) - c44 # Assuming c13+c44 >= 0
		return HexagonalMaterial(c11,c12,c13,c33,c44),ρ

# ----- Thomsen geometric materials -----

# If one is only interested in the speed of the pressure wave in the medium, 
# which defines the TTI geometric metric, then some of the Thomsen coefficients can be discarded.
# We refer to the geometric data as a Thomsen geometric material.


TGM_data=namedtuple('ThomsenGeometricMaterial_data',['Vp','Vs','ε','δ'])
class ThomsenGeometricMaterial(TGM_data):
	@classmethod
	def units(cls): return cls('m/s','m/s',1,1)
	
	@classmethod
	def from_c(cls,c11,c13,c33,c44,ρ=1):
		"Thomsen coefficients transformation (c11,c13,c33,c44) -> (Vp,Vs,ε,δ)"
		Vp = np.sqrt(c33/ρ)
		Vs = np.sqrt(c44/ρ)
		ε = (c11-c33)/(2*c33)
		δ = ((c13+c44)**2-(c33-c44)**2)/(2*c33*(c33-c44))
		return cls(Vp,Vs,ε,δ)
	
	def to_c(self):
		"Thomsen coefficients transformation (Vp,Vs,ε,δ) -> (c11,c13,c33,c44)"
		Vp,Vs,ε,δ = self
		c33 = Vp**2
		c44 = Vs**2
		c11 = c33*(2*ϵ+1)
		c13 = np.sqrt(2*c33*(c33-c44)*δ+(c33-c44)**2)-c44
		return c11,c13,c33,c44

	@classmethod
	def from_Elastic(cls,thomsenElasticMaterial):
		"Extracts the geometric coefficients (relevant for the pressure wave velocity)"
		Vp,Vs,ε,_,δ,_,_ = thomsenElasticMaterial
		return cls(Vp,Vs,ε,δ)

# ----- Some routines based on the minimal description c11,c13,c33,c44 -----


def is_definite(c11,c13,c33,c44): 
	"""Wether the coefficients define a positive definite hooke tensor, hence a convex inner sheet."""
	return (c11>0) & (c33>0) & (c11*c33>c13**2) & (c44>0) 

def is_separable(c11,c13,c33,c44): 
	"""Wether the inner and outer sheets are well separated"""
	return (c11>c44) & (c33>c44) & (c11*c33>c13**2) & (c44>0) & (c13+c44>0)

def is_second_sheet_convex(c11,c13,c33,c44):
	"""
	Wether the second sheet defined by the coefficients is convex.
	Formula obtained using formal computing software. 
	No proof, but checked on Thomsen's data and some examples by hand.

	"""
	det = c13**2 - c11*c33 + 4*c13*c44 + 4*c44**2
	slope_z = -c13**2 + c11*c33 - 2*c13*c44 - c44*(c33 + c44)
	slope_x = -c13**2 + c11*(c33 - c44) - 2*c13*c44 - c44**2
	inflexion = -c13**6 + 4*c11**3*c33**2*(c33 - c44) - 6*c13**5*c44 + 2*c11*c33*(7*c33 - 9*c44)*c44**3 - c33**2*c44**4 + c13**4*(6*c11*c33 - 2*c11*c44 - 2*c33*c44 - 9*c44**2) + 4*c13**3*c44*(6*c11*c33 - 2*c11*c44 - 2*c33*c44 + c44**2) - c11**2*c44*(4*c33**3 + 13*c33**2*c44 - 14*c33*c44**2 + c44**3) - 2*c13*c44*(c33*(c33 - 2*c44)*c44**2 + c11**2*(-3*c33 + c44)**2 - 2*c11*c44*(3*c33**2 - 4*c33*c44 + c44**2)) -  c13**2*(c11**2*(-3*c33 + c44)**2 + c44**2*(c33**2 + 6*c33*c44 - 12*c44**2) + 2*c11*c44*(-3*c33**2 - 8*c33*c44 + 3*c44**2))
#	print(f"{det=}, {slope_z=}, {slope_x=}, {inflexion=}")
	return np.where(det>0, (slope_x>0) & (slope_z>0), inflexion<0)


# ----- Thomsen tabulated data -----

TEM = ThomsenElasticMaterial

ThomsenData = OrderedDict([
	("Taylor sandstone",                     TEM(3368,1829,0.110,-0.127,-0.035,0.255,2.500)),
	("Mesaverde (4903) mudshale",            TEM(4529,2703,0.034,0.250,0.211,0.046,2.520)),
	("Mesaverde (4912) immature sandstone",  TEM(4476,2814,0.097,0.051,0.091,0.051,2.500)),
	("Mesaverde (4946) immature sandstone",  TEM(4099,2346,0.077,-0.039,0.010,0.066,2.450)),
	("Mesaverde (5469.5) silty sandstone",   TEM(4972,2899,0.056,-0.041,-0.003,0.067,2.630)),
	("Mesaverde (5481.3) immature sandstone",TEM(4349,2571,0.091,0.134,0.148,0.105,2.460)),
	("Mesaverde (5501) clayshale",           TEM(3928,2055,0.334,0.818,0.730,0.575,2.590)),
	("Mesaverde (5555.5) immature sandstone",TEM(4539,2706,0.060,0.147,0.143,0.045,2.480)),
	("Mesaverde (5566.3) laminated siltstone",TEM(4449,2585,0.091,0.688,0.565,0.046,2.570)),
	("Mesaverde (5837.5) immature sandstone",TEM(4672,2833,0.023,-0.013,0.002,0.013,2.470)),
	("Mesaverde (5858.6) clayshale",         TEM(3794,2074,0.189,0.154,0.204,0.175,2.560)),
	("Mesaverde (6423.6) calcareous sandstone",TEM(5460,3219,0.000,-0.345,-0.264,-0.007,2.690)),
	("Mesaverde (6455.1) immature sandstone",TEM(4418,2587,0.053,0.173,0.158,0.133,2.450)),
	("Mesaverde (6542.6) immature sandstone",TEM(4405,2542,0.080,-0.057,-0.003,0.093,2.510)),
	("Mesaverde (6563.7) mudshale",          TEM(5073,2998,0.010,0.009,0.012,-0.005,2.680)),
	("Mesaverde (7888.4) sandstone",         TEM(4869,2911,0.033,0.030,0.040,-0.019,2500)),
	("Mesaverde (7939.5) mudshale",          TEM(4296,2471,0.081,0.118,0.129,0.048,2.660)),

	("Mesaverde shale (350)",                TEM(3383,2438,0.065,-0.003,0.059,0.071,2.35)),
	("Mesaverde sandstone (1582)",           TEM(3688,2774,0.081,0.010,0.057,0.000,2.73)),
	("Mesaverde shale (1599)",               TEM(3901,2682,0.137,-0.078,-0.012,0.026,2.64)),
	("Mesaverde sandstone (1958)",           TEM(4237,3018,0.036,-0.037,-0.039,0.030,2.69)),
	("Mesaverde shale (1968)",               TEM(4846,3170,0.063,-0.031,0.008,0.028,2.69)),
	("Mesaverde sandstone (3512)",           TEM(4633,3231,-0.026,-0.004,-0.033,0.035,2.71)),
	("Mesaverde shale (3511)",               TEM(4359,3048,0.172,-0.088,0.000,0.157,2.81)),
	("Mesaverde sandstone (3805)",           TEM(3962,2926,0.055,-0.066,-0.089,0.041,2.87)),
	("Mesaverde shale (3883)",               TEM(3749,2621,0.128,-0.025,0.078,0.100,2.92)),
	("Dog Creek shale",                      TEM(1875,826,0.225,-0.020,0.100,0.345,2.000)),
	("Wills Point shale - 1",                TEM(1058,387,0.215,0.359,0.315,0.280,1.800)),
	("Wills Point shale - 2",                TEM(4130,2380,0.085,0.104,0.120,0.185,2640)),
	("Cotton Valley shale",                  TEM(4721,2890,0.135,0.172,0.205,0.180,2.640)),
	("Pierre shale - 1",                     TEM(2074,869,0.110,0.058,0.090,0.165,2.25)), # rho ?
	("Pierre shale - 2",                     TEM(2106,887,0.195,0.128,0.175,0.300,2.25)), # rho ?
	("Pierre shale - 3",                     TEM(2202,969,0.015,0.085,0.060,0.030,2.25)), # rho ?
	("shale (5000) - 1",                     TEM(3048,1490,0.255,-0.270,-0.050,0.480,2.420)),

	("shale (5000) - 2",                     TEM(3377,1490,0.200,-0.282,-0.075,0.510,2.420)),
	("Oil Shale",                            TEM(4231,2539,0.200,0.000,0.100,0.145,2.370)),
	("Green River shale - 1",                TEM(4167,2432,0.040,-0.013,0.010,0.145,2.370)),
	("Green River shale - 2",                TEM(4404,2582,0.025,0.056,0.055,0.020,2.310)),
	("Berea sandstone - 1",                  TEM(4206,2664,0.002,0.023,0.020,0.005,2.140)),
	("Berea sandstone - 2",                  TEM(3810,2368,0.030,0.037,0.045,0.030,2.160)),
	("Green River shale - 3",                TEM(3292,1768,0.195,-0.45,-0.220,0.180,2.075)),
	("Lance sandstone",                      TEM(5029,2987,-0.005,-0.032,-0.015,0.005,2.430)),
	("Ft. Union siltstone",                  TEM(4877,2941,0.045,-0.071,-0.045,0.040,2.600)),
	("Timber Mtn tuff",                      TEM(4846,1856,0.020,-0.003,-0.030,0.105,2.330)),
	("Muscovite crystal",                    TEM(4420,2091,1.12,-1.23,-0.235,2.28,2.79)),
	("Quartz crystal (hexag. approx.)",      TEM(6096,4481,-0.096,0.169,0.273,-0.159,2.65)),
	("Calcite crystal (hexag. approx.)",     TEM(5334,3353,0.369,0.127,0.579,0.169,2.71)),
	("Biotite crystal",                      TEM(4054,1341,1.222,-1.437,-0.388,6.12,3.05)),
	("Apatite crystal",                      TEM(6340,4389,0.097,0.257,0.586,0.079,3.218)),
	("Ice I crystal",                        TEM(3627,1676,-0.038,-0.10,-0.164,0.031,1.064)),
	("Aluminium-lucite composite",           TEM(2868,1350,0.97,-0.89,-0.09,1.30,1.86)),

	("Sandstone-shale",                      TEM(3009,1654,0.013,-0.010,-0.001,0.035,2.34)),
	("SS-anisotropic shale",                 TEM(3009,1654,0.059,-0.042,-0.001,0.163,2.34)),
	("Limestone-shale",                      TEM(3306,1819,0.134,-0.094,0.000,0.156,2.44)),
	("LS-anisotropic shale",                 TEM(3306,1819,0.169,-0.123,0.000,0.271,2.44)),
	("Anisotropic shale",                    TEM(2745,1508,0.103,-0.073,-0.001,0.345,2.34)),
	("Gas sand-water sand",                  TEM(1409,780,0.022,-0.002,0.018,0.004,2.03)),
	("Gypsum-weathered material",            TEM(1911,795,1.161,-1.075,-0.140,2.781,2.35))
])

# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

"""
This module gathers a few helper functions for plotting data, that are used throughout the 
illustrative notebooks.
"""

from os import path
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from . import LinearParallel as lp
import io



def SetTitle3D(ax,title):
	ax.text2D(0.5,0.95,title,transform=ax.transAxes,horizontalalignment='center')

def savefig(fig,fileName,dirName=None,ax=None,**kwargs):
	"""Save a figure:
	- in a given directory, possibly set in the properties of the function. 
	 Silently fails if dirName is None
	- with defaulted arguments, possibly set in the properties of the function
	"""
	# Choose the subplot to be saved 
	if ax is not None:
		kwargs['bbox_inches'] = ax.get_tightbbox(
			fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())

	# Set arguments to be passed
	for key,value in vars(savefig).items():
		if key not in kwargs and key!='dirName':
			kwargs[key]=value

	# Set directory
	if dirName is None: 
		if savefig.dirName is None: return 
		else: dirName=savefig.dirName

	# Save figure
	if path.isdir(dirName):
		fig.savefig(path.join(dirName,fileName),**kwargs) 
	else:
		print("savefig error: No such directory", dirName)
#		raise OSError(2, 'No such directory', dirName)

savefig.dirName = None
savefig.bbox_inches = 'tight'
savefig.pad_inches = 0
savefig.dpi = 300

def open_local_or_web(func,filepath,local_prefix="../",
	# Using data stored on the gh-pages (github pages) branch
	web_prefix='https://mirebeau.github.io/AdaptiveGridDiscretizations',
	web_suffix=''):
	try: return func(local_prefix+filepath)
	except FileNotFoundError:
		try : return func(filepath)  # Retry without the local prefix...
		except FileNotFoundError:
			import urllib
			return func(urllib.request.urlopen(web_prefix+filepath+web_suffix))
#https://www.tutorialspoint.com/how-to-open-an-image-from-the-url-in-pil
#https://stackoverflow.com/questions/8779197/how-to-link-files-directly-from-github-raw-github-com

def imread(*args,**kwargs):
	"""
	Reads the image into a numpy array. Tries to find it locally and on the web.
	- *args,**args : passed to open_local_or_web
	"""
	import PIL
	return np.array(open_local_or_web(PIL.Image.open,*args,**kwargs))

def animation_curve(X,Y,**kwargs):
	"""Animates a sequence of curves Y[0],Y[1],... with X as horizontal axis"""
	fig, ax = plt.subplots(); plt.close()
	ax.set_xlim(( X[0], X[-1]))
	ax.set_ylim(( np.min(Y), np.max(Y)))
	line, = ax.plot([], [])
	def func(i,Y): line.set_data(X,Y[i])
	kwargs.setdefault('interval',20)
	kwargs.setdefault('repeat',False)
	return animation.FuncAnimation(fig,func,fargs=(Y,),frames=len(Y),**kwargs)

# ---- Vectors fields, metrics ----

def quiver(X,Y,U,V,subsampling=tuple(),**kwargs):
	"""
	Pyplot quiver with additional arg:
	- subsampling (tuple or int). Subsample X,Y,U,V	
	"""
	if np.ndim(subsampling)==0: subsampling = (subsampling,)*2
	where = tuple(slice(None,None,s) for s in subsampling)
	def f(Z): return Z.__getitem__(where)
	return plt.quiver(f(X),f(Y),f(U),f(V),**kwargs)

def Tissot(metric,X,nθ=100,subsampling=5,scale=-1):
	"""
	Display the collection of unit balls of a two dimensional metric, also known as the 
	Tissot indicatrix.
	Inputs : 
	- metric : the metric to display
	- X : the geometric domain
	- nθ : number of angular directions
	- subsampling (integer or pair of integers): only display a subset of the unit balls
	- scale : scaling factor for the unit balls (if negative, then relative to auto scale)
	"""
	if subsampling is not None:
		if np.ndim(subsampling)==0: subsampling=[subsampling,subsampling]
		metric.set_interpolation(X)
		X = X[:,(subsampling[0]//2)::subsampling[0],(subsampling[1]//2)::subsampling[1]]
		metric = metric.at(X)

	dx = X[:,1,1]-X[:,0,0]
	θ = np.linspace(0,2*np.pi,nθ)
	U = np.array([np.cos(θ),np.sin(θ)]) # unit vectors
	bd = np.array([u[:,None,None] / metric.norm(u) for u in U.T])
	
	if scale<0:
		default_scale = 0.4*min(dx[0]/np.max(bd[:,0]), dx[1]/np.max(bd[:,1]))
		scale = np.abs(scale)*default_scale
	bd = (bd*scale + X).reshape((nθ,2,-1))
	plt.plot(bd[:,0],bd[:,1],color='red')
	plt.scatter(*X,s=1,color='black')
	return scale


# -------------- Array to image conversion ----------

def imshow_ij(image,**kwargs): 
	"""Show an image, using Cartesian array coordinates, 
	as with the option indexing='ij' of np.mesgrid."""
	return plt.imshow(np.moveaxis(image,0,1),origin='lower',**kwargs)
	
def arr2fig(image,xsize=None,**kwargs):
	"""
	Create a figure displaying the given image, 
	and nothing else. Uses Cartesian array coordinates.
	"""
	xshape,yshape = image.shape[:2]
	if xsize is None: xsize = min(6.4,4.8*xshape/yshape)
	ysize = xsize*yshape/xshape
	fig = plt.figure(figsize=[xsize,ysize])
	plt.axis('off')
	fig.tight_layout(pad=0)
	imshow_ij(image,**kwargs)
	return fig

def fig2arr(fig,shape,noalpha=True):
	"""
	Save the figure as an array with the given shape, 
	which must be proportional to its size. Uses Cartesian array coords.

	Approximate inverse of arr2fig.
	"""
	size = fig.get_size_inches()
	assert np.allclose(shape[0]*size[1],shape[1]*size[0])
		
	io_buf = io.BytesIO()
	fig.savefig(io_buf, format='raw',dpi=shape[0]/size[0])
	io_buf.seek(0)
	img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
					newshape=(shape[1],shape[0], -1))
	io_buf.close()
	return np.moveaxis(img_arr,0,1)[:,::-1,:(3 if noalpha else 4)]/255


# ---------- Interactive picking of points in image ----------

def pick_lines(n=np.inf,broken=True,arrow=False):
	"""Interactively pick some (broken) lines."""
	# Set default : [np.round(line).astype(int).tolist() for line in lines]
	plt.title(f"Pick points, type enter once\n to make a line, twice to terminate") 
	plt.show()
	
	lines = []
	while len(lines)<n:
		pts = np.array(plt.ginput(-1 if broken else 2))
		if len(pts)==0: break # Empty input means end
		elif len(pts)==1: continue # Invalid input
		pts = pts.T
		lines.append(pts)
		plt.plot(*pts,color='r')
		if arrow: plt.arrow(*pts[:,-1],*0.001*(pts[:,-1]-pts[:,-2]),color='r',head_width=10)
	
	plt.close()
	return lines

def pick_points(n=np.inf):
	"""Interactively pick some point coordinates."""
	plt.title(f"Pick {n} point(s)" if n<np.inf else 
		"Pick points, middle click or type enter to terminate") 
	plt.show()
	pts = np.array(plt.ginput(-1 if n==np.inf else n)).T
	plt.close()
	return pts

def input_default(prompt='',default=''):
	if default!='': prompt += f" (default={default}) "
	return input(prompt) or default


# ----- Convex body 3D display -----

def plotly_primal_dual_bodies(V):
	"""	
	Output : facet indices, facet measures. 
	"""
	import scipy.spatial
	import plotly.graph_objects as go

	# Use a convex hull routine to triangulate the primal convex body
	primal_body = scipy.spatial.ConvexHull(V.T) 

	Vsize = V.shape[1]
	if primal_body.vertices.size!=Vsize:  # Then primal_body.vertices == np.arange(Xsize)
		raise ValueError("Non-convex primal body ! See",set(range(Vsize))-set(primal_body.vertices))

	# Use counter-clockwise orientation for all triangles
	S = primal_body.simplices.T
	N = primal_body.neighbors.T
	cw = np.sign(lp.det(V[:,S]))==-1 
	S[1,cw],S[2,cw] = S[2,cw],S[1,cw] 
	N[1,cw],N[2,cw] = N[2,cw],N[1,cw] 

	# --- Plotly primal mesh object
	x,y,z = V; i,j,k = S;
	primal_mesh = go.Mesh3d(x=x,y=y,z=z, i=i,j=j,k=k)
	# ---

	# --- Plotly primal edges object
	V0=V[:,S]; V1=V[:,np.roll(S,1,axis=0)];
	xe,ye,ze = np.moveaxis([V0,V1,np.full_like(V0,np.nan)],0,1)
	primal_edges = go.Scatter3d(x=xe.T.flat,y=ye.T.flat,z=ze.T.flat,mode='lines')
	# ---
	
	Sg = lp.solve_AV(lp.transpose(V[:,S]),np.ones(S.shape)) # The vertices of the dual convex body
	Ng = Sg[:,N] # Gradient of the neighbor cells
	
	# --- Plotly dual edges object
	xe,ye,ze = np.moveaxis(np.array([Sg[:,None]+0*Ng,Ng,np.full_like(Ng,np.nan)]),0,1)
	dual_edges = go.Scatter3d(x=xe.T.flat,y=ye.T.flat,z=ze.T.flat,mode='lines')
	# ---
	
	# Choose a reference simplex for each vertex, hence a reference point for each dual facet
	XS = np.full(Vsize,-1,S.dtype)
	XS[S] = np.arange(S.shape[1],dtype=S.dtype)[None]
		
	# --- Plotly dual mesh object, using triangulated faces
	x,y,z = Sg
	i = XS[S]; j=np.roll(N,-1,axis=0); k=np.tile(np.arange(Sg.shape[1]),(3,1))
	dual_mesh=go.Mesh3d(x=x,y=y,z=z, i=i.flat,j=j.flat,k=k.flat)
	# ---
	
	return (primal_mesh,primal_edges),(dual_mesh,dual_edges)
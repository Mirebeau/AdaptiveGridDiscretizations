# Inspired by https://fr.slideshare.net/jimarlow
import json
from IPython.display import display, Markdown, HTML

"""
Builds and checks a table of contents for a notebook, or a section of notebooks.

Example usage in a notebook: 
from Miscellaneous import TocTools; print(TocTools.displayTOC('Prox_BeckmanOT','Div'))
"""


def MakeLink(inFName,volume):
	dirName = "../Notebooks_"+volume+"/"; extension = ".ipynb"
	print("Notebook ["+inFName+"]("+dirName+inFName+extension+") "
		+ ", from volume "+ volume + " [Summary]("+dirName+"Summary"+extension+") " )

#**Acknowledgement.** The experiments presented in these notebooks are part of ongoing research, 
#some of it with PhD student Guillaume Bonnet, in co-direction with Frederic Bonnans, 
#and PhD student François Desquilbet, in co-direction with Ludovic Métivier.

def Info(volume):
	if volume in ['NonDiv','Div','Algo','Repro']:
		return """
**Acknowledgement.** Some of the experiments presented in these notebooks are part of 
ongoing research with Ludovic Métivier and Da Chen.

Copyright Jean-Marie Mirebeau, Centre Borelli, ENS Paris-Saclay, CNRS, University Paris-Saclay
"""
	elif volume == 'FMM':
		return """
This Python&reg; notebook is intended as documentation and testing for the [HamiltonFastMarching (HFM) library](https://github.com/mirebeau/HamiltonFastMarching), which also has interfaces to the Matlab&reg; and Mathematica&reg; languages. 
More information on the HFM library in the manuscript:
* Jean-Marie Mirebeau, Jorg Portegies, "Hamiltonian Fast Marching: A numerical solver for anisotropic and non-holonomic eikonal PDEs", 2019 [(link)](https://hal.archives-ouvertes.fr/hal-01778322)

Copyright Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
"""
	elif volume == 'GPU':
		return """
**Acknowledgement.** The experiments presented in these notebooks are part of ongoing research.
The author would like to acknowledge fruitful informal discussions with L. Gayraud on the 
topic of GPU coding and optimization.

Copyright Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
"""

VolumeTitle = {
'FMM':"Fast Marching Methods",
'NonDiv':"Non-Divergence form PDEs",
'Div':"Divergence form PDEs",
'Algo':"Algorithmic tools",
'Repro':"Reproducible research",
'GPU':"GPU accelerated methods",
}


VolumeFilenames = {
'FMM':[
#Part : Isotropic and anisotropic metrics
    "Isotropic","Riemannian","Rander","AsymmetricQuadratic", 
#Part : Non holonomic metrics and curvature penalization
    "Curvature","Curvature3","DeviationHorizontality","Trailers",
#Part : Algorithmic enhancements to the fast marching method
    "Geodesics","HighAccuracy","Sensitivity","SensitivitySL","DistanceFromBoundary", #"SmartIO",
#Part : Motion planning
	"ClosedPaths","DubinsZermelo","BoatRouting","RadarModels","Interactive_CurvatureObstacles",
#Part : Seismology and crystallography
	"Seismic","TTI",
#Part : Image models and segmentation
	"Illusion","Tubular","Interactive_ConvexRegionSegmentation",
#Part : Other applications
    "FisherRao","MedialAxis",
#Part : Custom optimal control models, discrete states
	"DubinsState","ElasticaVariants",
],
'NonDiv':[
#Part : One space dimension
	"MonotoneSchemes1D","Time1D_NonDiv","OTBoundary1D",
#Part : Monotone numerical schemes # Second order non-linear PDEs
	"LinearMonotoneSchemes2D","NonlinearMonotoneFirst2D",
	"NonlinearMonotoneSecond2D","MongeAmpere", 
#Part : Eikonal equation and variants
	"EikonalEulerian","ShapeFromShading", 
#Part : Time dependent optimal control
	"BoatRoutingGeneric_Time","BoatRouting_Time", 
],
'Div':[
#Part : One space dimension
	"Time1D_Div",
#Part : Static problems
	"Elliptic","EllipticAsymmetric",
#Part : Linear elasticity
	"ElasticEnergy","ElasticWave",
#Part : Primal dual optimization
	"Prox_MinCut",
#Part : Applications
	"VaradhanGeodesics","AnisotropicDiffusion",
],
'Algo':[
#Part : Tensor decomposition techniques
	"TensorSelling","TensorVoronoi","TensorVoronoi6",
#Part : Generalized acuteness
	"SternBrocot","VoronoiVectors","SeismicNorm",
#Part : Automatic differentiation
	"Dense","Sparse","Reverse","ADBugs",
#Part : Domain representation
	"SubsetRd","FiniteDifferences",
#Part : Convex functions and convex bodies
	"Meissner",
],
"Repro":[
"PucciMongeAmpere","EikonalRate",
"Isotropic_GPU","Riemann_GPU","Rander_GPU","Curvature_GPU","Flow_GPU",
"Seismic_GPU","Walls_GPU","EikonalAD_GPU",
]
}

RepositoryDescription = """**Github repository** to run and modify the examples on your computer.
[AdaptiveGridDiscretizations](https://github.com/Mirebeau/AdaptiveGridDiscretizations)\n
"""

def CheckChapter(next,prev,identifier):
	if prev is None: prev = "## 0."
	def split(chap):
		indent,number = chap.split()[:2]
		if len(indent)==2:
			if number[-1]=='.':
				number = number[:-1]
		return indent,number
#		assert number[-1]=='.'
		
	indent,number = split(prev)
	nextPossibilities = [
	("#"*(len(indent) -i), number[:-(1+2*i)]+str(int(number[-(1+2*i)])+1) ) 
	for i in range(len(indent)-1)]
	nextPossibilities.append((indent+"#",number+".1"))

	if split(next) not in nextPossibilities:
		print(f"Error : {next} does not follow {prev}, see {identifier}") #, considered {nextPossibilities}")

	nextIndent,nextNumber = next.split()[:2]



def displayTOC(inFName,volume):
	with open(inFName+".ipynb", encoding='utf8') as data_file:
		data = json.load(data_file)
	contents = []
	prevChapter = None
	for c in data['cells']:
		s=c['source']
		if len(s)==0:
			continue
		line1 = s[0].strip()
		if line1.startswith('#'):
			count = line1.count('#')-1
			plainText = line1[count+1:].strip()
			if plainText[0].isdigit() and int(plainText[0])!=0:
				CheckChapter(line1,prevChapter,inFName)
				prevChapter=line1
				link = plainText.replace(' ','-')
				listItem = "  "*count + "* [" + plainText + "](#" + link + ")"
				contents.append(listItem)

	contents = ["[**Summary**](Summary.ipynb) of volume " + VolumeTitle[volume] + ", this series of notebooks.\n",
	"""[**Main summary**](../Summary.ipynb) of the Adaptive Grid Discretizations 
	book of notebooks, including the other volumes.\n""",
	"# Table of contents"] + contents + ["\n\n"+Info(volume)]

	return "\n".join(contents)
	

# 	display(Markdown("[**Summary**](Summary.ipynb) of this series of notebooks. "))
# 	display(Markdown("""[**Main summary**](../Summary.ipynb), including the other volumes of this work. """))
# #	display(HTML("<a id = 'table_of_contents'></a>"))
# 	display(Markdown("\n# Table of contents"))
# 	display(Markdown("\n".join(contents)))
# 	display(Markdown("\n\n"+Info(volume)))

def displayTOCs(volume,subdir=""):
	inFNames = VolumeFilenames[volume]
	contents = []
	part = ""
	part_counter = 0
	part_numerals = "ABCDEFGHIJK"
	chapter_counter = 0
	chapter_numerals = ["I","II","III","IV","V","VI","VII","VIII","IX","X"]
	for _inFName in inFNames:
		inFName = _inFName+".ipynb"
		with open(subdir+inFName, encoding='utf8') as data_file:
			data = json.load(data_file)
			# Display the chapter
			s=data['cells'][0]['source']
			sec = s[2][len("## Part : "):]
			if sec!=part:
				part=sec
				contents.append("### "+part_numerals[part_counter]+". "+part)
				part_counter+=1
				chapter_counter=0
			else:
				chapter_counter+=1
			chapter = s[3][len("## Chapter : "):].strip()
			contents.append(" " + "* "+chapter_numerals[chapter_counter] +
				". [" + chapter + "](" + inFName + ")")
			# Display the sub chapters
			for c in data['cells']:
				s = c['source']
				if len(s)==0: 
					continue
				line1 = s[0].strip()
				if line1.startswith('##') and line1[3].isdigit() and int(line1[3])!=0:
					contents.append(" "*2 + line1[len("## "):])
			contents.append("\n")

	contents = [RepositoryDescription,"# Table of contents",
		"[**Main summary**](../Summary.ipynb), including the other volumes of this work. "]+contents;
	return "\n".join(contents)
#	display(Markdown(RepositoryDescription))
#	display(Markdown("# Table of contents"))
#	display(Markdown("""[**Main summary**](../Summary.ipynb), including the other volumes of this work. """ ))
#	display(Markdown("\n".join(contents)))

def displayTOCss():
	extension = '.ipynb'
	contents = []
	volume_numerals = "1234"
	part_numerals = "ABCDEFGHIJK"
	chapter_numerals = ["I","II","III","IV","V","VI","VII","VIII","IX","X"]
	for volume_counter,volume in enumerate(['FMM','NonDiv','Div','Algo']):
		dirName = 'Notebooks_'+volume+'/'
		part = ""
		part_counter = 0

		inFName = dirName+'Summary'+extension
		with open(inFName, encoding='utf8') as data_file:
			data = json.load(data_file)
			s = data['cells'][0]['source']
			volumeTitle = s[2][len("# Volume : "):]
			contents.append("### " + volume_numerals[volume_counter]+". "+
				"["+volumeTitle+"]("+inFName+")")

		# Display parts and chapters
		for _inFName in VolumeFilenames[volume]:
			inFName = dirName+_inFName+extension
			with open(inFName, encoding='utf8') as data_file:
				data = json.load(data_file)
				# Display the chapter
				s=data['cells'][0]['source']
				sec = s[2][len("## Part : "):]
				if sec!=part:
					part=sec
					contents.append(" * "+part_numerals[part_counter]+". "+part)
					part_counter+=1
					chapter_counter=0
				else:
					chapter_counter+=1
				chapter = s[3][len("## Chapter : "):].strip()
				contents.append("  " + "* "+chapter_numerals[chapter_counter] +
					". [" + chapter + "](" + inFName + ")")
		contents.append("")

	contents = ["# Table of contents"]+contents
	return "\n".join(contents)
#	display(Markdown("# Table of contents"))
#	display(Markdown("\n".join(contents)))








# Code automatically exported from notebook ElasticEnergy.ipynb in directory Notebooks_Div
# Do not modify
from ... import LinearParallel as lp
from ... import FiniteDifferences as fd
from agd.Metrics.Seismic import Hooke
from ... import AutomaticDifferentiation as ad
from ... import Domain
from agd.Plotting import savefig; #savefig.dirName = 'Images/ElasticityDirichlet'

norm_infinity = ad.Optimization.norm_infinity
norm_average = ad.Optimization.norm_average
mica,_ = Hooke.mica # Hooke tensor associated to this crystal


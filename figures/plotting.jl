using PyPlot, PyCall, Printf
@pyimport scipy.optimize as opt
@pyimport matplotlib as mpl
@pyimport matplotlib.gridspec as gspec
PyPlot.matplotlib[:rcParams]["font.family"] = "serif"
PyPlot.matplotlib[:rcParams]["mathtext.fontset"] = "dejavuserif"
PyPlot.matplotlib[:rcParams]["font.size"] = 30
PyPlot.matplotlib[:rcParams]["xtick.labelsize"] = 22
PyPlot.matplotlib[:rcParams]["legend.fancybox"] = false
PyPlot.rc("font",   size=22)
PyPlot.rc("axes",   titlesize=22)
PyPlot.rc("xtick",  labelsize=22)
PyPlot.rc("ytick",  labelsize=22)
PyPlot.rc("axes",   labelsize=22)
PyPlot.rc("legend", fontsize=16)
PyPlot.rc("figure", titlesize=22)
PyPlot.rc("text",   usetex=true)
rc("font", family="serif")
colors = PyPlot.matplotlib[:rcParams]["axes.prop_cycle"][:by_key]()["color"]
slc = (lo,hi) -> pycall(pybuiltin("slice"), PyObject, lo, hi)
meshgrid(x,y) = (x=repeat(x',outer=(length(y),1)), y=repeat(y,outer=(1,length(x))))

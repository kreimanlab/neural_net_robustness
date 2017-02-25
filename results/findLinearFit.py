#!/usr/bin/python3

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from glob import glob
import pickle
import numpy as np
import pandas as pd

from scipy.stats import ranksums,mannwhitneyu,linregress

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rc('axes', labelsize=14) 
matplotlib.rc('ytick', labelsize=14) 
matplotlib.rc('xtick', labelsize=14) 

colorList = sns.color_palette()+sns.color_palette("husl", 8)[:2]+sns.color_palette("Set2", 8)

###############################################################
# plot settings

metric = "top5performance"
# metric = "top1performance"
# perturbationName = "mutateGaussian"
# perturbationName = "nodeKnockout"
perturbationName = "synapseKnockout"
networkName = "alexnet"
# networkName = "vgg16"
# maxPercent = 1.0
maxPercent = 9999
# statOnly = True
statOnly = False

folderName = "/home/nick/projects/neural_net_robustness/results/ILSVRC2012/val/perturbations/"

##############################################################

data = pd.DataFrame(columns=('layer', 'percent', 'trial','performance'))

fileList = glob(folderName+"/"+networkName+'*'+perturbationName+"*")
for fileName in fileList:
	tmp = pickle.load(open(fileName,"rb"))
	performance = (tmp[metric])
	layerName = fileName.split(networkName+"-")[1].split("-")[0]
	percent = fileName.split(perturbationName)[1].split("-")[0]
	trialNum = fileName.split("-")[-1].split(".p")[0]
	if float(percent) <= maxPercent:
		data.loc[len(data)] = [layerName,percent,trialNum,performance]

layerNameList = sorted(list(set(data["layer"].values)))

for thisLayerName in layerNameList:
	tmp = pickle.load(open("/home/nick/projects/neural_net_robustness/results/ILSVRC2012/val/"+networkName+".p","rb"))
	performance = (tmp[metric])
	layerName = thisLayerName
	percent = "0.0"
	trialNum = "1"
	data.loc[len(data)] = [layerName,percent,trialNum,performance]


noPerturbation = "0.0"
minPerturbation = "0.30"
maxPerturbation = "0.70"

slope, intercept, r_value, p_value, std_err = linregress(np.array(data[data["layer"]!="conv_1"]["percent"].values,dtype=float),data[data["layer"]!="conv_1"]["performance"].values) 

print( "r^2:",r_value**2)


slope, intercept, r_value, p_value, std_err = linregress(np.array(data["percent"].values,dtype=float),data["performance"].values) 

print( "including conv_1:",r_value**2)

slope, intercept, r_value, p_value, std_err = linregress(np.array(data[data["layer"]=="conv_1"]["percent"].values,dtype=float),data[data["layer"]=="conv_1"]["performance"].values) 

print( "only conv_1:",r_value**2)

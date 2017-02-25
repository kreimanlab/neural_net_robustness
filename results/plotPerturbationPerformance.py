#!/usr/bin/python3

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from glob import glob
import pickle
import numpy as np
import pandas as pd

from scipy.stats import ranksums,mannwhitneyu

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rc('axes', labelsize=14) 
matplotlib.rc('ytick', labelsize=14) 
matplotlib.rc('xtick', labelsize=14) 

colorList = sns.color_palette()+sns.color_palette("husl", 8)[:2]+sns.color_palette("Set2", 8)

##########################################################
# plot settings

metric = "top5performance"
# metric = "top1performance"
perturbationName = "mutateGaussian"
# perturbationName = "nodeKnockout"
# perturbationName = "synapseKnockout"
networkName = "alexnet"
# networkName = "vgg16"
# maxPercent = 1.0
maxPercent = 9999
# maxPercent = 0.9
statOnly = True
# statOnly = False

folderName = "/home/nick/projects/neural_net_robustness/results/ILSVRC2012/val/perturbations/"

noPerturbation = "0.0"
minPerturbation = "0.10"
maxPerturbation = "2.00"

###########################################################

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


if statOnly:
	for thisLayerName in layerNameList:
		layerData = data[data["layer"]==thisLayerName]
		smallPerturbation = layerData[layerData["percent"]==maxPerturbation]["performance"].values
		largePerturbation = layerData[layerData["percent"]==maxPerturbation]["performance"].values
		print (thisLayerName, "fall in performance:", ranksums(smallPerturbation,largePerturbation)[1]) 
	print()

	for thisLayerName in layerNameList:
		layerData = data[data["layer"]==thisLayerName]
		largePerturbation = layerData[layerData["percent"]==maxPerturbation]["performance"].values
		print (thisLayerName, maxPerturbation,"performance:",np.mean(largePerturbation), "--" , (np.mean(largePerturbation)-np.mean(data[data["percent"]==noPerturbation]["performance"].values))/np.mean(data[data["percent"]==noPerturbation]["performance"].values),"%")
	print()

	c = 0
	for i in range(0,len(layerNameList)):
		if "conv" in layerNameList[i]:
			layerData1 = data[data["layer"]==layerNameList[i]]
			performance1 = layerData1[layerData1["percent"]==maxPerturbation]["performance"].values

			for j in range(0,i):
				if "conv" in layerNameList[j]:
					layerData2 = data[data["layer"]==layerNameList[j]]
					performance2 = layerData2[layerData2["percent"]==maxPerturbation]["performance"].values
							
					print ("comare",layerNameList[i],"(",np.mean(performance1),")",layerNameList[j],"(",np.mean(performance2),")",ranksums(performance1,performance2)[1], "--", [layerNameList[i],layerNameList[j]][int(np.mean(performance1)<np.mean(performance2))],"higher performance", np.mean(performance1)<np.mean(performance2))
					c+=1

		print()
	print (c)

	data.to_csv(networkName+"_"+perturbationName+"_"+metric+".csv")

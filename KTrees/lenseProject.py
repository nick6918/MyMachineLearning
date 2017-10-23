from ktrees import createTree
from numpy import array
from plotTree import createPlot

def getData(filename):
	fp = open(filename)
	lenses = array([line.strip().split("\t") for line in fp.readlines()]).T
	return lenses


featureNameList = ['age', 'prescript', 'astigmatic', 'tearRate']
lenses = getData("lenses.txt")
lenseTree = createTree(lenses, featureNameList)
createPlot(lenseTree)



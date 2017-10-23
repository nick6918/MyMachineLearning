import matplotlib.pyplot as plt

decisionNode = dict(boxstyle = 'sawtooth', fc='0.8')
leafNode = dict(boxstyle = 'round4', fc='0.8')
arrow_args = dict(arrowstyle="<-")

def getExampleTree():
	listOfTrees = [{'no surface': {0: 'No', 1: {'flippers':{0: 'No', 1:'Yes'}}}}, {'no surface': {0: 'No', 1: {'flippers': {0: {'head': {0: 'No', 1: 'Yes'}}, 1:'Yes'}}}}]
	return listOfTrees

def plotNode(nodetxt, parent, center, nodeType):
	#require use to achieve createPlot.ax1
	createPlot.ax1.annotate(nodetxt, xy=parent, xycoords='axes fraction', xytext=center, textcoords='axes fraction', \
		va = 'center', ha='center', bbox=nodeType, arrowprops=arrow_args)

def getNumLeafs(myTree):
	numberofLeaf = 0
	if type(myTree) != type(dict()):
		return 1
	firstFeatureName = myTree.keys()[0]
	for featureValue in myTree[firstFeatureName].keys():
		numberofLeaf += getNumLeafs(myTree[firstFeatureName][featureValue])
	return numberofLeaf

def getTreeDepth(myTree):
	if type(myTree) != type(dict()):
		return 1
	maxSubDepth = 0
	firstFeatureName = myTree.keys()[0]
	for featureValue in myTree[firstFeatureName].keys():
		currentDepth = getTreeDepth(myTree[firstFeatureName][featureValue])
		if currentDepth > maxSubDepth:
			maxSubDepth = currentDepth
	return 1 + maxSubDepth

def plotMidText(ctr, parent, text):
	xMid = (parent[0] - ctr[0])/2.0 + ctr[0]
	yMid = (parent[1] - ctr[1])/2.0 + ctr[1]
	createPlot.ax1.text(xMid, yMid, text)

def plotTree(myTree, parent, nodetxt):
	#require user to achieve createPlot.ax1, plotTree.totalW , plotTree.totalD , plotTree.xoff , plotTree.yoff
	numLeafs = getNumLeafs(myTree)
	depth = getTreeDepth(myTree)-1
	firstFeatureName = myTree.keys()[0]
	ctrPoint = (plotTree.xoff + (1.0 + numLeafs*1.0)/2.0/plotTree.totalW, plotTree.yoff)
	plotMidText(ctrPoint, parent, nodetxt)
	plotNode(firstFeatureName, parent, ctrPoint, decisionNode)
	secondTreeList = myTree[firstFeatureName]
	plotTree.yoff = plotTree.yoff - 1.0/plotTree.totalD
	for key in secondTreeList.keys():
		if type(secondTreeList[key]) == type(dict()):
			plotTree(secondTreeList[key], ctrPoint, str(key))
		else:
			plotTree.xoff = plotTree.xoff + 1.0/plotTree.totalW
			plotNode(secondTreeList[key], ctrPoint, (plotTree.xoff, plotTree.yoff), leafNode)
			plotMidText((plotTree.xoff, plotTree.yoff), ctrPoint, str(key))
	plotTree.yoff = plotTree.yoff + 1.0/plotTree.totalD

def createPlot(myTree):
	fig = plt.figure()
	fig.clf()
	# Test 1, plot random arrow
	# createPlot.ax1 = plt.subplot(111)
	# plotNode('DecisionNode', (0.5, 0.1), (0.1, 0.5), decisionNode)
	# plotNode('leafNode', (0.8, 0.1), (0.3, 0.8), leafNode)
	# Test 2, plot a tree
	axprops = dict(xticks=[], yticks=[])
	createPlot.ax1 = plt.subplot(111, **axprops)
	plotTree.totalW = float(getNumLeafs(myTree))
	plotTree.totalD = float(getTreeDepth(myTree))
	plotTree.xoff = -0.5/plotTree.totalW
	plotTree.yoff = 1.0
	plotTree(myTree, (0.5, 1.0), ' ')
	plt.show()

if __name__ == '__main__':
	trees = getExampleTree()
	myTree = trees[1]
	createPlot(myTree)

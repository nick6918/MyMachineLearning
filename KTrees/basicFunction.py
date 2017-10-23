from numpy import array, newaxis, zeros
import matplotlib.pyplot  as plt 
import random

def createLabeledData(count, featureCount, labelcount, min, max):

	datalist = []
	labels =[]
	for i in range(count):
		currentData = []
		for j in range(featureCount):
			currentData.append(random.uniform(min, max))
		datalist.append(currentData)
		labels.append(random.randint(0, labelcount))
	return array(datalist).T, array([labels]).T

def autoNorm(data):

	minVals = data.min(axis=1)[..., newaxis]
	maxVals = data.max(axis=1)[..., newaxis]
	ranges = maxVals - minVals
	return (data*1.0 - minVals) / ranges

def file2Matrix(filename, featureNumber):
	fp = open(filename)
	arrayOfLines = fp.readlines()
	numberOfLines = len(arrayOfLines)
	returnMax = zeros((numberOfLines, featureNumber))
	index = 0
	classLabel = []
	for line in arrayOfLines:
		line = line.strip()
		listFromLine = line.split('\t')
		returnMax[index, :] = listFromLine[0:featureNumber]
		classLabel.append(listFromLine[featureNumber])
		index += 1
	return returnMax.T, array([classLabel]).T
	
if __name__ == '__main__':
		
	def test():
		print "****************create1DExampleData*****************"
		first = createExampleData()
		print first[0]
		print first[1]
		print "****************createLabeledData*****************"
		second = createLabeledData(4, 10, 3, 0, 2)
		print second[0]
		print second[1]
		print "****************autoNorm*****************"
		print autoNorm(first[0])
		print "****************file2Matrix*****************"
		data, label = file2Matrix('datingTestSet.txt', 3)
		fig =plt.figure()
		sbfig = fig.add_subplot(111)
		sbfig.scatter(data[1], data[2])
		plt.show()
	test()
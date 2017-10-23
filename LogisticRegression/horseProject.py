from logisticRegression import logisticClassify, stocgredientAscent, plotRegression
from numpy import array, ones, newaxis, delete
from matplotlib.pyplot import figure

def collectData(trainFile, testFile):
	fp = open(trainFile)
	trainData = []
	testData = []
	for line in fp.readlines():
		lineArr = line.strip().split('\t')
		trainData.append([float(item) for item in lineArr])
	trainMatrix = array(trainData).T
	trainLabel = trainMatrix[-1]
	delete(trainMatrix, [len(trainMatrix)-1], axis=1)
	fp.close()
	fp = open(testFile)
	for line in fp.readlines():
		lineArr = line.strip().split('\t')
		testData.append([float(item) for item in lineArr])
	testMatrix = array(testData).T
	testLabel = testMatrix[-1]
	delete(testMatrix, [len(testMatrix)-1], axis=1)
	fp.close()
	return trainMatrix, trainLabel, testMatrix, testLabel

trainMatrix, trainLabel, testMatrix, testLabel = collectData("horseColicTraining.txt", "horseColicTest.txt")

fig = figure()
ax = fig.add_subplot(111)
ax.scatter(tr)

initials = ones((len(trainMatrix), 1))

parameters = stocgredientAscent(trainMatrix, trainLabel, initials, 0.01, 500)
bestLabel = logisticClassify(parameters, testMatrix)
errorCount = 0
totalCount = len(bestLabel)
for i in range(totalCount):
	if bestLabel[i] != testLabel[i]:
		print i, "th data got wrong label, correct: ", testLabel[i], " predict: ", bestLabel[i]
		errorCount += 1
print totalCount, " tests has been run, errorRate: ", errorCount*1.0/totalCount
#plotRegression(trainMatrix, trainLabel, parameters)


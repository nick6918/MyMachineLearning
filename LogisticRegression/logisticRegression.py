from numpy import array, exp, ones, dot, vstack, arange, newaxis, shape, zeros
import random

def loadDataSet():
	dataMatrix = []
	labelList = []
	fp = open('testSet.txt', "r")
	lines = fp.readlines()
	fp.close()
	for line in lines:
		line = line.strip().split()
		dataMatrix.append([1.0, float(line[0]), float(line[1])])
		labelList.append(int(line[2]))
	return array(dataMatrix).T, array(labelList)

def sigmond(z):
	return 1.0/(1+exp(-z))

def dparameters(parameters, x_matrix, y_vector):
	dataNumber = len(x_matrix[0])
	#hx for (1, n)
	delta_vector = dot(parameters.T, x_matrix) - y_vector
	#return (m, 1)
	dparameters = dot(x_matrix, delta_vector.T)
	return dparameters

def dparametersSigmond(parameters, x_matrix, y_vector):
	#add 1 for x0

	delta_vector = y_vector - sigmond(dot(parameters.T, x_matrix))
	dparameters = dot(x_matrix, delta_vector.T)
	return dparameters


def gredientAscent(x_matrix, y_vector, initialParameters, step, times):
	#initialParameters = ones((featureNumber, 1))
	parameters = initialParameters
	for i in range(times):
		delta_vector = y_vector - sigmond(dot(parameters.T, x_matrix))
		dparameters = dot(x_matrix, delta_vector.T)
		parameters = parameters + step * dparameters
	return parameters

def stocgredientAscent(x_matrix, y_vector, initialParameters, step, iter_times):
	#initialParameters = ones((featureNumber, 1))
	parameters = initialParameters
	dataNumbers = len(x_matrix[0])
	para_matrix = zeros((len(parameters), iter_times))
	for j in range(iter_times):			#modification 1
		dataIndex = range(dataNumbers)
		for i in range(dataNumbers):
			alpha = 4.0/(1+i+j) + step		#modification 2
			#alpha = step #comparison for 2

			index = random.randint(0, len(dataIndex)-1)  #modification 3
			data_index = dataIndex[index]
			del(dataIndex[index])

			#data_index = i      #comparison for modification 3
			delta = y_vector[data_index] - sigmond(dot(parameters.T, x_matrix[..., data_index]))
			dparameters = delta * x_matrix[..., data_index]
			parameters = parameters + alpha * dparameters[..., newaxis]
		para_matrix[..., j:j+1] = parameters
		if j%100 == 0:
			print j, " times have finished"
	#plot_parameters(para_matrix)
	return parameters

def logisticClassify(parameters, test_matrix):
	result = sigmond(dot(parameters.T, test_matrix))
	return [int(item > 0.5) for item in result[0]]

def plot_parameters(para_matrix):
	import matplotlib.pyplot as plt
	fig = plt.figure()
	for j in range(len(parameters)):
		ax = fig.add_subplot(311+j)
		ax.plot(range(len(para_matrix[0])), para_matrix[j])
	plt.show()
	return 0

def plotRegression(x_matrix, y_vector, parameters):
	#label must be int
	COLOR = ["red", "green", "gray", "blue", "yellow"]
	import matplotlib.pyplot as plt
	labelSet = set(y_vector)
	xCord = {}
	yCord = {}
	for i in range(len(x_matrix[0])):
		currentLabel = int(y_vector[i])
		if currentLabel in xCord.keys():
			xCord[currentLabel].append(x_matrix[1, i])
			yCord[currentLabel].append(x_matrix[2, i])
		else:
			xCord[currentLabel]=list()
			yCord[currentLabel]=list()
	index = 0
	for label in labelSet:
		ax.scatter(xCord[label], yCord[label], s=30, c=COLOR[index])
		index += 1
	x = arange(-3.0, 3, 0.1)
	y = (-parameters[0]-parameters[1]*x)/parameters[2]
	ax.plot(x, y)
	plt.show()
	return 0

if __name__ == '__main__':
	x_matrix, y_vector = loadDataSet()
	parameters = ones((len(x_matrix), 1))
	#parameters = gredientAscent(x_matrix, y_vector, parameters, 0.001, 500)
	parameters = stocgredientAscent(x_matrix, y_vector, parameters, 0.001, 200)
	plotRegression(x_matrix, y_vector, parameters)


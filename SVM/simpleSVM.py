from numpy import array, delete, dot, ones, zeros, newaxis, nonzero
import random
from copy import copy
from math import ceil

class Data:
	def __init__(self, dataMatrix, labelList, C, epsilon):
		self.dataMatrix = dataMatrix
		self.labelList = labelList
		self.C = C
		self.epsilon = epsilon
		self.ecache = ones((len(dataMatrix[0]), 2))
		self.alphas = zeros((len(dataMatrix[0]), 1))
		self.b = 0

def loadDataSet(filename):
	dataList = []
	fp = open(filename)
	for line in fp.readlines():
		lineArr = line.strip().split("\t")
		dataList.append([float(item) for item in lineArr])
	fp.close()
	dataMatrix = array(dataList).T
	labelList = dataMatrix[-1]
	delete(dataMatrix, [len(dataMatrix), ], axis = 1)
	return dataMatrix, labelList

def selectRandomSecondIndex(firstIndex, m):
	#require: m is an integer
	index = firstIndex
	while(index == firstIndex):
		index = random.randint(0, m-1)
	return index

def clipAlpha(aj, H, L):
	#to limit alpha inside [L, H]
	if aj > H:
		aj = H
	if L > aj:
		aj = L
	return aj

def calcEi(i, data):
	xi = data.dataMatrix[..., i:i+1]
	gxi = dot(dot(xi.T, data.dataMatrix), data.alphas * data.labelList[..., newaxis]) + data.b
	Ei = gxi - data.labelList[i]
	return Ei

def update(data, k):
	Ei = calcEi(k, data)
	data.ecache[k] = [1, Ei]

def innerLoop(i, data, kernelFunc):
	#get everything
	dataMatrix = data.dataMatrix
	labelList = data.labelList
	C = data.C
	alphas = data.alphas
	b = data.b
	epsilon = data.epsilon

	#for each iteration, we use xi to find another xj, optimize it if possible
	xi = dataMatrix[..., i:i+1]
	yi = labelList[i]
	#gxi = wx+b = simga alphajyj <xi, xj>
	Ei = calcEi(i, data)
	#check if it doesn't qualify in KKT
	if ((yi*Ei < -epsilon) and (alphas[i] < C)) or ((yi*Ei > epsilon) and alphas[i] > 0):
		#NOT quaified, choose it
		j = selectBestJ(i, Ei, data, len(dataMatrix[0]))
		xj = dataMatrix[..., j]
		yj = labelList[j]
		gxj = dot(dot(xj.T, dataMatrix), alphas * labelList[..., newaxis]) + b
		Ej = calcEi(j, data)
		alphaJold = alphas[j, 0].copy()
		alphaIold = alphas[i, 0].copy()

		if (yi == yj):
			L = max(alphas[j]+alphas[i]-C, 0)
			H = min(C, alphas[i]+alphas[j])
		else:
			L = max(0, alphas[j]-alphas[i])
			H = min(C, C+alphas[j]-alphas[j])

		if (L == H): 
			print "L==H"
			return 0

		eta = -(kernelFunc(dataMatrix, i, i) - kernelFunc(dataMatrix, j, j) + 2*kernelFunc(dataMatrix, i, j))
		if eta >= 0:
			print "eta >= 0"
			return 0

		alphaJnew = alphaJold - yj * (Ei - Ej) / eta
		alphaJnew = clipAlpha(alphaJnew, H, L)
		update(data, j)
		if abs(alphaJnew - alphaJold) < 0.0001:
			print "alpha j not moving enough"
			return 0
		update(data, i)
		alphaInew = alphaIold + yi*yj*(alphaJold - alphaJnew)
		alphas[j, 0] = alphaJnew
		alphas[i, 0] = alphaInew

		print "alpha updated: ", alphas.T

		#update b
		bnew1 = b - Ei - yi*kernelFunc(dataMatrix, i, i)*(alphaInew - alphaIold) - yj*kernelFunc(dataMatrix, i, j)*(alphaJnew - alphaJold)
		bnew2 = b - Ej - yi*kernelFunc(dataMatrix, i, j)*(alphaInew - alphaIold) - yj*kernelFunc(dataMatrix, j, j)*(alphaJnew - alphaJold)
		if alphaJnew > C or alphaJnew < 0:
			b = bnew1
		elif alphaInew > C or alphaInew < 0:
			b = bnew2
		else:
			b = (bnew1+bnew2)/2
		print "This round: i: %d, j:%d" % ( i, j)
		data.dataMatrix = dataMatrix
		data.labelList = labelList
		data.alphas = alphas
		data.b = b
		return 1
	else:
		print "%d already correctly classfied, pass" % (i)
		return 0

def plattSMO(dataMatrix, labelList, C, epsilon, maxIteration, kernelFunc):
	featureNumber, dataNumber = dataMatrix.shape
	data = Data(dataMatrix, labelList, C, epsilon)
	it = 0
	entireSet = True
	alphaPairsChanged = 0
	while(it < maxIteration and ((alphaPairsChanged > 0) or entireSet)):
		alphaPairsChanged = 0
		if entireSet:
			for i in range(dataNumber):
				alphaPairsChanged += innerLoop(i, data, kernelFunc)
			it += 1
		else:
			optimizableIndex = nonzero(data.alphas[(data.alphas > 0) * (data.alphas < C)])[0]
			for i in optimizableIndex:
				alphaPairsChanged += innerLoop(i, data, kernelFunc)
			it += 1
		if entireSet: 
			entireSet = False
		elif alphaPairsChanged == 0:
			entireSet = True
	return data.alphas, data.b



def simpleSMO(dataMatrix, labelList, C, epsilon, maxIteration, kernelFunc):
	featureNumber, dataNumber = dataMatrix.shape
	alphas = ones((dataNumber, 1))
	b = 0
	it = 0
	while(it < maxIteration):
		it += 1
		alphaPairsChanged = 0
		for i in range(dataNumber):
			#for each iteration, we use xi to find another xj, optimize it if possible
			xi = dataMatrix[..., i:i+1]
			yi = labelList[i]
			#gxi = wx+b = simga alphajyj <xi, xj>
			Ei = calcEi(i, dataMatrix, labelList, alphas, b)
			#check if it doesn't qualify in KKT
			if ((yi*Ei < -epsilon) and (alphas[i] < C)) or ((yi*Ei > epsilon) and alphas[i] > 0):
				#NOT quaified, choose it
				j = selectRandomSecondIndex(i, dataNumber)
				xj = dataMatrix[..., j]
				yj = labelList[j]
				gxj = dot(dot(xj.T, dataMatrix), alphas * labelList[..., newaxis]) + b
				Ej = calcEi(j, dataMatrix, labelList, alphas, b)
				alphaJold = alphas[j, 0].copy()
				alphaIold = alphas[i, 0].copy()

				if (yi == yj):
					L = max(alphas[j]+alphas[i]-C, 0)
					H = min(C, alphas[i]+alphas[j])
				else:
					L = max(0, alphas[j]-alphas[i])
					H = min(C, C+alphas[j]-alphas[j])

				if (L == H): 
					print "L==H"
					continue

				eta = -(kernelFunc(dataMatrix, i, i) - kernelFunc(dataMatrix, j, j) + 2*kernelFunc(dataMatrix, i, j))
				if eta >= 0:
					print "eta >= 0"
					continue

				alphaJnew = alphaJold - yj * (Ei - Ej) / eta
				alphaJnew = clipAlpha(alphaJnew, H, L)
				if abs(alphaJnew - alphaJold) < 0.0001:
					print "alpha j not moving enough"
					continue
				alphaInew = alphaIold + yi*yj*(alphaJold - alphaJnew)
				alphas[j, 0] = alphaJnew
				alphas[i, 0] = alphaInew

				#update b
				bnew1 = b - Ei - yi*kernelFunc(dataMatrix, i, i)*(alphaInew - alphaIold) - yj*kernelFunc(dataMatrix, i, j)*(alphaJnew - alphaJold)
				bnew2 = b - Ej - yi*kernelFunc(dataMatrix, i, j)*(alphaInew - alphaIold) - yj*kernelFunc(dataMatrix, j, j)*(alphaJnew - alphaJold)
				if alphaJnew > C or alphaJnew < 0:
					b = bnew1
				elif alphaInew > C or alphaInew < 0:
					b = bnew2
				else:
					b = (bnew1+bnew2)/2
				alphaPairsChanged += 1
				print "iter: %d i: %d, j:%d, pairChanged: %d" % (it, i, j, alphaPairsChanged)
			else:
				print "%d already correctly classfied, pass" % (i)
		if alphaPairsChanged == 0:
			it += 1
		else:
			it = 0
		print "iter times: %d, %d pairs changed this iteration" % (it, alphaPairsChanged)
	return b, alphas

def selectBestJ(i, Ei, data, N):
	#cache: the first row represent where 0<alpha<C, heuristic alg
	validEiList = nonzero(data.ecache[..., 0])[0]
	maxDelta = 0
	maxIndex = None
	maxEj = 0
	data.ecache[i] = [1, Ei]
	for index in validEiList:
		if index == i:
			continue
		Ej = calcEi(index, data)
		delta = abs(Ei - Ej)
		if delta > maxDelta:
			maxDelta = delta
			maxIndex = index
			maxEj = Ej
	if not maxIndex:
		maxIndex = selectRandomSecondIndex(i, N)
		maxEj = calcEi(maxIndex, data)
	return maxIndex



def basicKernelFunc(dataMatrix, i, j):
	#inner cross
	return dot(dataMatrix[..., i].T, dataMatrix[..., j])

if __name__ == '__main__':

	dataMatrix, labelList = loadDataSet('testSet.txt')
	#b, alphas = simpleSMO(dataMatrix, labelList, 0.6, 0.001, 40, basicKernelFunc)
	alphas, b = plattSMO(dataMatrix, labelList, 0.6, 0.001, 40, basicKernelFunc)	
	print b
	print alphas[alphas>0]





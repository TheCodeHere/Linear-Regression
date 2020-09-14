import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax

def GetData():
	file = open("blood_pressure.txt","r")

	data = []
	for line in file:
		data.append(list(map(float,line.split())))
	data = np.array(data)

	#Center the data
	mean = np.mean(data, axis=0) #centrar
	mean[0] = 0
	data -= mean

	print("Data:\n", data)

	# Dividing Data (data, target)
	X = data[:, :-1]
	y = data[:, -1:]

	return X,y


def Gen_Sintetic_Data(numPoints = 100, bias=1, variance=10):
	X = []
	y = []

	# Generating a straight line
	for i in range(0, numPoints):
		# Include bias
		X.append([1.0,float(i)])

		# Objective
		y.append([(i + bias) + np.random.uniform(0, 1) * variance])

	X = np.array(X)
	y = np.array(y)

	return X,y

def Ploting(tit = "default"):
	# add grid
	plt.grid(True,linestyle='--')

	# add legend
	#plt.legend(loc='upper left')
	plt.tight_layout()
	plt.title(tit)

	# add x,y axes labels
	plt.xlabel('X1')
	plt.ylabel('X2')
	

def Can_LinReg(X,Y):
	print("##########################  Canonical Linear Regression ##########################")

	# Constant
	m, n = X.shape

	# computing theta parameters
	Theta = np.linalg.pinv(np.dot(X.T,X))
	Theta = np.dot(Theta,X.T)
	Theta = np.dot(Theta,Y)

	# Calculate Cost
	Error = np.dot(X, Theta) - Y
	# Avoid large quantities
	Error /= 1
	J = np.dot(Error.T, Error) / (2 * m)

	print("Final cost value: ", *J[0])

	print("\nFinal Theta value:\n", Theta, "\n")

	# plot line
	eje_x = np.linspace(X[:,1].min(),X[:,1].max(),100)
	plt.plot(eje_x, Theta[0] + Theta[1] * eje_x, '--r', label='Lin Reg')
	Ploting("Canonical Linear Regression")

def Can_Reg_LinReg(X,Y, lamb = 100):
	print("##########################  Canonical Linear Regression + Regularization  ##########################")

	#Constant
	m, n = X.shape

	# regularization factor
	diag_Mat = np.diag([0] + [lamb]*(n-1))

	# computing theta parameters
	Theta = np.linalg.pinv(np.dot(X.T,X) + diag_Mat)
	Theta = np.dot(Theta,X.T)
	Theta = np.dot(Theta,Y)

	# Calculate Cost
	Error = np.dot(X, Theta) - Y
	# Avoid large quantities
	Error /= 1
	J = np.dot(Error.T, Error) / (2 * m)
	J += (lamb / (2 * m)) * np.dot(Theta[1:].T, Theta[1:])

	print("Final cost value: ", *J[0])

	print("\nFinal Theta value:\n", Theta, "\n")

	# plot line
	eje_x = np.linspace(X[:,1].min(),X[:,1].max(),100)
	plt.plot(eje_x, Theta[0] + Theta[1] * eje_x, '--r', label='Lin Reg')
	Ploting("Canonical Linear Regression + Regularization")


def Grad_LinReg(X,Y,epochs = 500,alpha=0.001):
	print("##########################  Gradient Descent ##########################")

	# Constants
	m,n = X.shape
	Theta = np.random.rand(n, 1)

	cost = []
	for _ in range(epochs):
		Error = np.dot(X, Theta) - Y
		# Avoid large quantities
		Error /= 1

		# Calculating Cost
		J = np.dot(Error.T,Error) / (2 * m)
		cost.append(*J[0])
		#print("COST FUNCTION: ",cost[-1])

		# New Theta
		DJ = np.dot(X.T,Error)
		Theta = Theta - (alpha/m) * DJ

	print("Final Cost Function value: ", cost[-1])

	print("\nFinal Theta value:\n", Theta, "\n")

	# plot line aproximation
	eje_x = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
	plt.plot(eje_x, Theta[0] + Theta[1] * eje_x, '--r', label='Lin Reg')
	Ploting("Linear Regression Gradient")

	# plot the cost function
	plt.figure()
	plt.plot(list(range(epochs)), cost, '-r')
	Ploting("Cost Function - LinR Gradient")

def Grad_Reg_LinReg(X,Y,epochs=500,alpha=0.001,lamb=100):
	print("##########################  Gradient + Regularization ##########################")

	#constants
	m, n = X.shape
	Theta = np.random.rand(n, 1)
	diag_Mat = np.diag([1] + [1-(alpha/m)*lamb]*(n-1))

	cost = []
	for _ in range(epochs):
		Error = np.dot(X, Theta) - Y
		# Avoid large quantities
		Error /= 1

		# Cost Function
		J = np.dot(Error.T, Error) / (2*m)
		J += (lamb/(2*m))*np.dot(Theta[1:].T,Theta[1:])
		cost.append(*J[0])
		#print("COST FUNCTION: ", cost[-1])

		# New Theta
		DJ = np.dot(X.T, Error)
		Theta = np.dot(diag_Mat,Theta) - (alpha/m) * DJ

	print("Final Cost Function value: ", cost[-1])

	print("\nFinal Theta value:\n", Theta, "\n")

	# plot line aproximation
	eje_x = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
	plt.plot(eje_x, Theta[0] + Theta[1] * eje_x, '--r', label='Lin Reg')
	Ploting("Linear Regression Gradient + Regularization")

	# plot the cost function
	plt.figure()
	plt.plot(list(range(epochs)), cost, '-r')
	Ploting("Cost Function - LinR Gradient + Regularization")

if __name__ == '__main__':

	X,y = GetData()
	#X, y = Gen_Sintetic_Data()

	#print("X:\n", X, "\ny:\n", y)

	# Canonical Linear regression
	plt.figure()
	plt.scatter(X.T[1], y, s=35, marker='.')
	Can_LinReg(X,y)

	# Canonical Regularized Linear regression
	plt.figure()
	plt.scatter(X.T[1], y, s=35, marker='.')
	Can_Reg_LinReg(X, y)

	# Gradient descent Linear regression
	plt.figure()
	plt.scatter(X.T[1], y, s=35, marker='.')
	Grad_LinReg(X,y,epochs=300)

	# Regularized Gradient descent Linear regression
	plt.figure()
	plt.scatter(X.T[1], y, s=35, marker='.')
	Grad_Reg_LinReg(X,y,epochs=300)

	plt.show()
	

	

	
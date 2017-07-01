import numpy as np

class NeuralNetwork():
	def __init__(self):
		#Seed the random number generator
		np.random.seed(1)

		#model a single neuron, with 3 input connections and 1 output connection/
		#assign random weights to a 3x1 matric, with values in ther ande -1 to 1 and mean 0
		self.weights = {}
		self.num_layers = 1
		self.adjustments = {}

	def add_layer(self, shape):
		#Create weights with shape specified + biases
		self.weights[self.num_layers] = np.vstack((2*np.random.random(shape) - 1, 2* np.random.random((1,shape[1]))-1))
		#intialize the adjustments for these weights to zero
		self.adjustments[self.num_layers] = np.zeros(shape)
		self.num_layers += 1

	def __sigmoid(self, x):
			return 1/(1+np.exp(-x))

	def __sigmoid_deriviative(self,x):
			return x*(1-x)	

	def train(self, inputs, targets, num_iter, learning_rate = 1, stop_accuracy=1e-5):
		error = []
		for iteration in range(num_iter):
			for i in range(len(inputs)):
				x = inputs[i]
				y = targets[i]
				#pass the training set through our neural network
				output = self.__forward_propagate(x)

				#calculate error
				loss = self.sum_squared_error(output[self.num_layers], y)
				error.append(loss)

				#calculate adjustments
				self.__back_propagate(output,y)
			self.__gradient_descent(i,learning_rate)	

			#check if accuracy criterion is satisfied
			if np.mean(error[-(i+1):]) < stop_accuracy and iteration > 0:
				break

		return(np.asarray(error), iteration+1)		

		# for iteration in range(num_iter):
		# 	#pass the training set through the network
		# 	output = self.predict(inputs)

		# 	#calculate the error
		# 	error = targets - output

		# 	#multiply the error by the input and again by the gradient of the sigmoid curve
		# 	adjustment = dot(inputs.T, error * self.__sigmoid_deriviative(output))

		# 	#adjust the weights
		# 	self.weights += adjustment

	def predict(self, data):
		#pass data through pretrained network
		for layer in range(1, self.num_layers+1):
			data = np.dot(data, self.weights[layer-1][:,:-1]) + self.weights[layer-1][:,-1]
			data = self.__sigmoid(data)
		return data

	def __forward_propagate(self,data):
		#progagate through network and hold values for use in back-propagation
		activation_values = {}
		activation_values[1] = data
		for layer in range(2, self.num_layers+1):
			data = np.dot(data.T ,self.weights[layer-1][:-1,:]) + self.weights[layer-1][-1,:].T # + self.biases[layer]
			data = self.__sigmoid(data).T
			activation_values[layer] = data
		return activation_values

	def simple_error(self, outputs, targets):
		return targets - outputs

	def sum_squared_error(self, outputs, targets):
		return 0.5*np.mean(np.sum(np.power(outputs-targets,2), axis =1))

	def __back_propagate(self, output, target):
		deltas = {}
		#delta of output layer
		deltas[self.num_layers] = output[self.num_layers] - target

		#delta of hidden layers
		for layer in reversed(range(2,self.num_layers)): #all layers except input/output
			a_val = output[layer]
			weights = self.weights[layer][:-1, :]
			prev_deltas = deltas[layer+1]
			deltas[layer] = np.multiply(np.dot(weights,prev_deltas), self.__sigmoid_deriviative(a_val))

		#calculate total adjustments based on deltas
		for layer in range(1, self.num_layers):
			self.adjustments[layer] += np.dot(deltas[layer+1], output[layer].T).T

	def __gradient_descent(self, batch_size, learning_rate):
		#calculate partial dericative and takea  step in that direction
		for layer in range(1, self.num_layers):
			partial_d = (1/batch_size)*self.adjustments[layer]
			self.weights[layer][:-1,:] += learning_rate * -partial_d
			self.weights[layer][-1,:] += learning_rate*1e-3 * -partial_d[-1,:]


if __name__ == "__main__":

	#initialise a single neuron neural network
	nn = NeuralNetwork()

	#add layers to neural network
	nn.add_layer((2,9))
	nn.add_layer((9,1))

	print("Random starting synaptic weights: ") 
	print(nn.weights) 

	#The training set
	inputs = np.asarray([[0,0],[0,1],[1,0],[1,1]]).reshape(4,2,1)
	targets = np.asarray([[0], [1], [1], [0]])

	#train the neural network
	#10,000 iterations and adjust each time
	error, iteration = nn.train(inputs, targets, 10000)
	print("Error = ", np.mean(error[-4:]))
	print("Iterations needed to train = ", iteration)
	

	print("New synaptic weights after training:  ") 
	print(nn.weights) 

	#Test network
	print("predicting:") 
	#getting key error: 0 and idk why
	test_data = np.asarray([[0,0],[0,1],[1,0],[1,1]]).reshape(4,2,1)
	print(nn.predict(test_data)) 
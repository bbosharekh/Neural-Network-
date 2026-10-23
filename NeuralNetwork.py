class NeuralNetwork:
    def __init__(self,input_size=784, hidden_layers=[512, 512], output_size=10):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.weights = []
        self.biases = []

        # Input to hidden layers network 

        self.weights.append(0.01 * np.random.randn(input_size, hidden_layers[0]))
        self.biases.append(np.zeros((1, hidden_layers[0]))) 

        # Hidden layers network
        for i in range(len(hidden_layers)-1):
            self.weights.append(0.01 * np.random.randn(hidden_layers[i],hidden_layers[i+1]))
            self.biases.append(np.zeros((1, hidden_layers[i+1]))) 

        # Hidden layers network to output 
        self.weights.append(0.01 * np.random.randn(hidden_layers[len(hidden_layers)-1], output_size))
        self.biases.append(np.zeros((1,output_size))) 

    def relu(self, Z: np.ndarray) -> np.ndarray:
        # Applies the Rectified Linear Unit (ReLU) activation function. 
        return np.maximum(0, Z)
        
   def forward(self, X: np.ndarray) -> np.ndarray:
        # Store activations 
        current_activation = X

        # Applying ReLU to hidden layers
        for i in range(len(self.weights) - 1):
            Z = np.dot(current_activation, self.weights[i]) + self.biases[i]
            current_activation = self.relu(Z)
            
        # Output layer 
        final_Z = np.dot(current_activation, self.weights[-1]) + self.biases[-1]
        
        return final_Z


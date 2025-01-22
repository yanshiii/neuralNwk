***Neural Network for Handwritten Digit Recognition***

📌 Overview

This project implements a simple neural network for recognizing handwritten digits using the MNIST dataset. The model is built using NumPy and follows a fully connected feedforward architecture with ReLU activation and softmax output.

📌 Features

✅ Implements a 2-layer neural network from scratch

✅ Uses ReLU activation function for hidden layers and softmax for output

✅ Trains on the MNIST dataset

✅ Saves and loads trained model parameters using pickle

✅ Predicts and visualizes handwritten digits from the test dataset

📌 Installation

Clone the repository (Using Git Bash):

"git clone https://github.com/yanshiii/neuralNwk.git"

"cd neuralNwk"

Install dependencies:

"pip install numpy pandas matplotlib keras"

📌 Usage

🔹 Train the model: "python neural_network.py --train"

🔹 This will train the network and save the trained parameters in trained_params.pkl.

🔹 Test the model: "python neural_network.py --test"

🔹 The script will load the saved model and display predictions for test images.

📂 Files

neural_network.py - Main script for training and testing the neural network

trained_params.pkl - Serialized file storing trained weights and biases

📌 Model Architecture

Input Layer: 784 neurons (28x28 flattened image)

Hidden Layer: 10 neurons (ReLU activation)

Output Layer: 10 neurons (softmax activation)

📌 Example Output

After training, the model will display predicted vs. actual labels for selected test images.
![Screenshot 2025-01-22 115157](https://github.com/user-attachments/assets/46b18960-1930-453d-b1e7-2d4b40eafd8d)


🤝 Contributing

Feel free to fork the repository and submit pull requests to improve the project.

📜 License

This project is open-source and available under the MIT License.

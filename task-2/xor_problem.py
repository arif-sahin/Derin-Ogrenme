import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt



# Sigmoid and Derivative
def sigmoid(x):
        return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
        return x * (1 - x)


# Xor inputs and outputs
x = np.array([[0,0],[1,0], [0,1], [1,1]])
y = np.array([[0],[1],[1],[0]])


# Weights and biases
np.random.seed(0)
w1 = np.random.rand(2,2) # input to hidden layer, 2 inputs to 2 hidden
w2 = np.random.rand(2,1) # hiddent to output layer
b1 = np.random.rand(2) # random biases for hidden layer
b2 = np.random.rand(1) # random bias for output layer

lr = 0.1

for epoch in range(10000):
    # Feed Forward
    hidden_input = np.dot(x, w1) + b1
    hidden_output = sigmoid(hidden_input)
    output_input = np.dot(hidden_output, w2) + b2
    output = sigmoid(output_input)

    # Loss
    loss = -np.mean(y * np.log(output) + (1 - y) * np.log(1 - output))

    #Back propagation
    output_error = output - y
    output_delta = output_error * sigmoid_derivative(output)

    hidden_error = np.dot(output_delta, w2.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

    # Update weights and biases
    w1 -= lr * np.dot(x.T, hidden_delta)
    w2 -= lr * np.dot(hidden_output.T, output_delta)
    b1 -= lr * np.sum(hidden_delta, axis=0)
    b2 -= lr * np.sum(output_delta, axis=0)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")


predictions = (output > 0.5).astype(int) # Converting probabilites to binary
accuracy = np.mean(predictions == y) * 100
precision = precision_score(y, predictions, zero_division=0)
recall = recall_score(y, predictions, zero_division=0)
f1 = f1_score(y, predictions, zero_division=0)

print("Final output after training:")
print(output)
print(f"Accuracy: {accuracy:.2f}%")
print("Predictions:")
for i in range(len(x)):
    print(f"Input: {x[i]} => Predicted Output: {predictions[i]}, Actual Output: {y[i]}")


# Confusion Matrix
cm = confusion_matrix(y, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap='Greens')
plt.title("Confusion Matrix")
plt.show()

# Decision Boundary
fig, axes = plt.subplots(figsize=(7, 5))

# Create a mesh grid for the decision boundary
xx, yy = np.meshgrid(np.linspace(-0.2, 1.2, 300), np.linspace(-0.2, 1.2, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
hidden_input_grid = np.dot(grid, w1) + b1
hidden_output_grid = sigmoid(hidden_input_grid)
output_input_grid = np.dot(hidden_output_grid, w2) + b2
grid_output = sigmoid(output_input_grid)
grid_predictions = (grid_output > 0.5).astype(int)

# Contour Plot for Decision Boundary
contour = axes.contourf(xx, yy, grid_predictions.reshape(xx.shape), levels=50, cmap='RdYlGn', alpha=0.7)
contour_line = axes.contour(xx, yy, grid_predictions.reshape(xx.shape), levels=[0.5], colors='black', linewidths=2, linestyles='--')

# Scatter plot for input points
scatter = axes.scatter(x[:, 0], x[:, 1], c=y[:, 0], s=300, zorder=5, edgecolors='white', linewidth=2)

# Annotations for each point
for i, txt in enumerate(x):
    axes.annotate(f"({txt[0]}, {txt[1]}) => {y[i][0]}", 
                   (x[i, 0], x[i, 1]),
                   textcoords="offset points", 
                   xytext=(10, 10) if y[i] == 1 else (-15, -15),
                   ha='center', fontsize=10, color='white', fontweight='bold')

axes.set_title('Learned Decision Boundary', fontsize=13, fontweight='bold')
axes.set_xlabel('Input 1', fontsize=11)
axes.set_ylabel('Input 2', fontsize=11)
axes.grid()

plt.suptitle('Decision Boundary for XOR Problem', fontsize=13, fontweight='bold')
plt.colorbar(contour)
plt.tight_layout()
plt.show()
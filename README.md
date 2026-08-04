# Perceptron Classifier in Python

A from-scratch Python implementation of Rosenblatt's Perceptron, applied to a binary classification problem using the famous Iris dataset.

## 🧠 About the Project

This project demonstrates the fundamental concepts of machine learning by implementing the classic Perceptron algorithm (invented in 1957 by Frank Rosenblatt) using only standard numerical libraries.

The model is trained to distinguish between two flower species (*Iris-setosa* and *Iris-versicolor*) based on two features: sepal length and petal length.

### Features:
- Object-oriented Perceptron implementation (weights, bias, learning rate, and epochs).
- Automated dataset downloading directly from the UCI Machine Learning Repository.
- Two visualizations using Matplotlib:
  1. **Error Trend Plot:** Shows the number of misclassifications (updates) during each training epoch.
  2. **Decision Regions Plot:** Visualizes the linear decision boundary created by the algorithm to separate the two classes.

---

## 🧮 The Mathematical Model

Beyond the NumPy implementation, the Perceptron is built on a surprisingly simple mathematical foundation. Here's the theory behind the code.

### Net Input

Each training example is a feature vector $\mathbf{x}$ (here, sepal length and petal length). The Perceptron computes a weighted linear combination of these inputs, called the *net input*:

$$z = w_1 x_1 + w_2 x_2 + \dots + w_m x_m + b = \mathbf{w}^T \mathbf{x} + b$$

where $\mathbf{w}$ is the weight vector and $b$ is the bias unit. Geometrically, this expression defines a hyperplane in the feature space, but with only two features, simply a straight line in the plane.

### Activation Function (Unit Step)

The net input $z$ is passed through a step function (the *unit step function*), which determines the predicted class:

$$
\hat{y} =
\begin{cases}
1 & \text{if } z \geq 0 \\
0 & \text{if } z < 0
\end{cases}
$$

This is exactly what the `predict()` method implements.

### The Perceptron Learning Rule

The core of the algorithm is the weight update rule, introduced by Rosenblatt in 1957. For each training example $(x^{(i)}, y^{(i)})$, weights and bias are updated as follows:

$$w_j := w_j + \Delta w_j \quad \text{where} \quad \Delta w_j = \eta \left( y^{(i)} - \hat{y}^{(i)} \right) x_j^{(i)}$$

$$b := b + \Delta b \quad \text{where} \quad \Delta b = \eta \left( y^{(i)} - \hat{y}^{(i)} \right)$$

where $\eta$ (eta) is the *learning rate*. A few key properties worth noting:

- If the prediction is correct ($y^{(i)} = \hat{y}^{(i)}$), the update is zero: the weights don't change.
- If the prediction is wrong, the weights shift in the direction that reduces the error, proportionally to the input $x_j^{(i)}$ and to the learning rate.
- This is exactly what the `errors` counter tracks in the code: every time `update != 0`, a misclassification occurred, and the weights were corrected.

### Convergence

An interesting theoretical fact: the **Perceptron Convergence Theorem** guarantees that if the two classes are *linearly separable*, the algorithm is guaranteed to converge to a solution (a hyperplane that perfectly separates the classes) in a finite number of steps. This is visible in the `errors_` plot: as training progresses, the number of misclassifications per epoch should drop to zero if the data is linearly separable — which, for *Iris-setosa* vs. *Iris-versicolor* using these two features, it is.

If the classes were **not** linearly separable, the Perceptron would never converge and the weights would keep oscillating indefinitely — one of the historical limitations that motivated later models (like Adaline and multi-layer networks).

---

## 🎓 What I Learned

Working on this project helped me understand:

- How a linear binary classifier works from first principles, without relying on high-level ML frameworks.
- The connection between the **geometric interpretation** (a separating hyperplane) and the **algebraic update rule** (gradient-free, error-driven weight correction).
- Why the choice of learning rate $\eta$ affects convergence speed and stability.
- The practical limits of the Perceptron: it only works when the data is linearly separable, which is why it's the historical starting point before moving to models like Adaline (gradient descent) and eventually multi-layer perceptrons.

---

*Based on the book "Machine Learning with PyTorch and Scikit-Learn".*

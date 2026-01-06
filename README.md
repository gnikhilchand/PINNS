# Physics-Informed Neural Networks (PINNs) for Lid-Driven Cavity Flow

This repository contains an implementation of **Physics-Informed Neural Networks (PINNs)** to solve the steady-state **2D Navier-Stokes equations**. 

The project demonstrates a mesh-free deep learning approach to Computational Fluid Dynamics (CFD), specifically modeling the classic **Lid-Driven Cavity** problem without using conventional grid-based solvers (like FVM or FEM).

## 📄 Project Overview

In this project, a Deep Neural Network is trained to approximate the solution to the Navier-Stokes equations. The network takes spatial coordinates $(x, y)$ as input and outputs the flow parameters (Stream function $\psi$ and Pressure $p$).

Instead of training on labeled data (supervised learning), the network is trained by minimizing a loss function composed of:
1.  **Boundary Conditions:** Enforcing no-slip walls and the moving lid velocity.
2.  **Physics Residuals:** Minimizing the error in the governing partial differential equations (PDEs) at randomly sampled collocation points inside the domain.

### The Physics (Governing Equations)
The model solves the dimensionless Incompressible Navier-Stokes equations:

$$u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = -\frac{\partial p}{\partial x} + \frac{1}{Re} \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)$$

$$u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = -\frac{\partial p}{\partial y} + \frac{1}{Re} \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right)$$

Subject to the continuity equation:
$$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$$

*(Where $Re$ is the Reynolds Number, $u$ and $v$ are velocity components derived from the stream function $\psi$.)*

## 🛠️ Technologies Used

* **Python 3.x**
* **TensorFlow:** For building the neural network and utilizing **Automatic Differentiation (AutoGrad)** to compute spatial derivatives of the PDEs.
* **NumPy:** For data manipulation and domain discretization.
* **Matplotlib:** For visualizing the vector fields, pressure contours, and streamlines.


## 🚀 Getting Started

### Prerequisites
To run this notebook, you need the following libraries:

```bash
pip install tensorflow numpy matplotlib
```
Clone the repository:

```bash
git clone [https://github.com/gnikhilchand/PINNS.git](https://github.com/gnikhilchand/PINNS.git)
```
## 🧠 Methodology
### Network Architecture: A fully connected dense neural network (DNN).
### Collocation Points: The domain is sampled with random points where the PDE residuals are evaluated.
### Automatic Differentiation: TensorFlow's GradientTape is used to calculate exact derivatives of the neural network output with respect to input coordinates ($x, y$), allowing for the direct encoding of physics into the loss function.



```bash
pip install tensorflow numpy matplotlib

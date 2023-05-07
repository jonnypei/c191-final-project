# PHYS/CHEM/CS C191 Final Project: HHL Code Implementation

This github repository provides an implementation of the HHL algorithm (quantum method for solving systems of linear equations) described in the paper "Quantum algorithm for solving linear systems of equations" (https://arxiv.org/abs/0811.3171) by Aram W. Harrow, Avinatan Hassidim, and Seth Lloyd.

Contributors: Aditya Ramabadran, Jonny Pei, and Yuki Ito

References: https://learn.qiskit.org/course/ch-applications/solving-linear-systems-of-equations-using-hhl-and-its-qiskit-implementation 


## Getting Started

1. Install the latest version of Python. We developed our code using Python 3.9.7, but any slightly older/newer version should work fine.

2. Clone this repo and move into it by running:
```
git clone https://github.com/jonnypei/c191-final-project.git
cd c191-final-project
```

3. Install required packages by running:
```
pip install -r requirements.txt
```

4. Install the repository by running:
```
pip install -e .
```

## How to Use

We test our (naive) implementation of HHL in ```hhl_test.py``` and verify its correctness
against the classical linear systems solver. You can run the file
in the ```c191-final-project``` directory as follows:
```
python hhl_test.py
```
Feel free to play around with the inputs $A, \vec{b}$ to test out our implementation. 

**Note:** make sure $A \in \mathbb{R}^{N \times N}$ is hermitian and has dimension that is power of 2. If $A$ isn't hermitian, you can adjust your inputs to be as follows:

$$A^{\prime} = \begin{pmatrix}
0_{N \times N} & A \\
A^{\dagger} & 0_{N \times N}
\end{pmatrix}, \quad \vec{b}^{\prime} = \begin{pmatrix}
\vec{b} \\
\vec{0}
\end{pmatrix}$$

Then, HHL will solve the equation $A^{\prime} \vec{x}^{\prime} = \vec{b}^{\prime}$ to obtain 

$$\vec{x}^{\prime} = \begin{pmatrix}
\vec{0} \\
\vec{x}
\end{pmatrix}$$

You can then extract your desired solution using e.g. ```x = x_prime[N:]```.

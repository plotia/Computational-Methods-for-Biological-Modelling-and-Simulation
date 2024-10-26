# Optimization in Model Fitting: Learning Rate Parameters of a Biochemical Reaction Model

In this problem, we will explore a use of optimization in the context of model fitting. We will specifically try to learn rate parameters of a biochemical reaction model. Let us suppose that we are studying a reaction system:

\[ 2A + B \xrightleftharpoons[k_2]{k_1} C \]

with forward rate \( k_1 \) and reverse rate \( k_2 \). This can be described by a system of differential equations:

\[
\begin{align*}
\frac{dA}{dt} &= -2k_1A^2B + 2k_2C \\
\frac{dB}{dt} &= -k_1A^2B + k_2C \\
\frac{dC}{dt} &= k_1A^2B - k_2C
\end{align*}
\]

We will assume that we are given some measured concentrations of these molecules versus time and would like to learn the rate constants of the reaction.

## a. Forward Euler Simulator

We will first need a way to simulate the reaction system. Write pseudocode for a forward Euler simulator that takes starting concentrations \( A_0 \), \( B_0 \), and \( C_0 \), a step size \( \Delta t \), and a number of time steps \( n \) and produces a matrix of concentrations \( X \) of each of the reactants at each time point, where \( X_{1i} \) is the value of \( A(i\Delta t) \), \( X_{2i} \) is the value of \( B(i\Delta t) \), and \( X_{3i} \) is the value of \( C(i\Delta t) \).

### Pseudocode


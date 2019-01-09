<h2>MachineLearning-MultiLinearRegressionWithGradientDescent</h2>
<h3>Description:</h3>
<ul style="list-style-type:disc">
<li>Python script to estimate coefficients for multilinear regression using gradient descent algorithm. </li>
<li>Linear regression implemented from scratch.</li>
<li>Using simulated data of job prospects given AI and machine learning skills.</li>
</ul>

<p float="left">
  <img src="/linearRegressionCost.gif" width="400" alt="Cost of algorithm improvement through epochs."/>
  <img src="/linearRegressionFit.gif" width="460"alt="Shape of the hyperplane as cost from algorithm improves through epochs."/>
</p>

 


\begin{equation*}
  y = W^T X + \epsilon
\end{equation}        
<u>Where:</u><br>
y is the target,<br>
$w_0$ is the intercept (bias value)<br>
W is a vector parameters (weights) <strong>to be estimated</strong>.
  \begin{align*}
    W &= \begin{bmatrix}
           w_{0} \\
           w_{1} \\
           w_{2} \\
           \vdots \\
           w_{N}
         \end{bmatrix}
 \end{align}
X is a matrix of 1's and K feature weights and N data points of <strong>given inputs</strong>
  \begin{align*}
    X &= \begin{bmatrix}
1&x_{12}&\cdots &x_{1K} \\
1&x_{22}&\cdots &x_{2K} \\
\vdots & \vdots & \ddots & \vdots\\
1&x_{N2}&\cdots &x_{NK}
\end{bmatrix}
 \end{align}
 
 
 and $\epsilon$ is a vector of estimation errors denoted
  \begin{align*}
    W &= \begin{bmatrix}
           \epsilon_{1} \\
           \epsilon_{2} \\
           \vdots \\
           \epsilon_{N}
         \end{bmatrix}
 \end{align} 
 

<h3>Expected Outputt</h3>
<pre>
=======================================================================
MULTI LINEAR REGRESSION USING GRADIENT DESCENT TERMINATION RESULTS
=======================================================================
Initial Weights were:             0.0, 0.0, 0.0.
   With initial cost:          3281.9.
        # Iterations:       2,500,000.
       Final weights:    w0:+24.94, w1:+0.32, w2:+0.483.
          Final cost:            +8.1.
                RMSE:            +4.0, R-Squared:         +0.7
=======================================================================
Finished
</pre>

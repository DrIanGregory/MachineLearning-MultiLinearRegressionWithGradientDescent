<h2>MachineLearning-MultiLinearRegressionWithGradientDescent</h2>
<h3>Description:</h3>
<ul style="list-style-type:disc">
<li>Python script to estimate coefficients for multilinear regression using gradient descent algorithm. </li>
<li>Linear regression implemented from scratch.</li>
<li>Using simulated data of job prospects given AI and machine learning skills.</li>
</ul>

<p float="left">
  <img src="images/linearRegressionCost.gif" width="400" alt="Cost of algorithm improvement through epochs."/>
  <img src="images/linearRegressionFit.gif" width="460"alt="Shape of the hyperplane as cost from algorithm improves through epochs."/>
</p>

 

$y = W^T X + \epsilon$
$ax^2 + bx + c = 0$, then $x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$.


$$
  y = W^T X + \epsilon
$$

<u>Where:</u><br>
y is the target,<br>
<p>$w_0$ is the intercept (bias value)<p><br>
W is a vector parameters (weights) <strong>to be estimated</strong>.

$$
  \begin{align*}
    W &= \begin{bmatrix}
           w_{0} \\
           w_{1} \\
           w_{2} \\
           \vdots \\
           w_{N}
         \end{bmatrix}
 \end{align}
$$

 
X is a matrix of 1's and K feature weights and N data points of <strong>given inputs</strong>

$$
  \begin{align*}
    X &= \begin{bmatrix}
1&x_{12}&\cdots &x_{1K} \\
1&x_{22}&\cdots &x_{2K} \\
\vdots & \vdots & \ddots & \vdots\\
1&x_{N2}&\cdots &x_{NK}
\end{bmatrix}
 \end{align}
$$ 
 
 and $\epsilon$ is a vector of estimation errors denoted
 $$
  \begin{align*}
    \epsilon &= \begin{bmatrix}
           \epsilon_{1} \\
           \epsilon_{2} \\
           \vdots \\
           \epsilon_{N}
         \end{bmatrix}
 \end{align} 
 $$


 
<ul style="list-style-type:disc">
	<li>The loss function chosen is <strong>minimum mean square error</strong> given by:</li>
</ul>

$$
	\begin{equation*}\label{eq:MultipleLinearRegressionCostFunction}
		C(W) = \frac{1}{N} \sum^{N}_{n=1}(( W^T X_n)-y_n)^2
	\end{equation} 
$$

<ul style="list-style-type:disc">
	<li>With partial derivatives</li>
</ul>
$$
\begin{align*}
	\frac{\partial C}{\partial w_0} &= -\frac{2}{N} \sum_{n=1}^{N} ((W^T X_n ) - y_n) \\
	\frac{\partial C}{\partial w_i} &= -\frac{2}{N} \sum_{n=1}^{N} x_i((W^T X_n - y_n)) 
\end{align}
$$

<ul style="list-style-type:disc">
 <li>With weight updates given by:</li>
</ul>

$$
\begin{equation*}
    w_n = w_n - \alpha \frac{\partial C}{\partial w_n}
\end{equation}
$$

<ul style="list-style-type:disc">
	<li>Where $\alpha$ is the <em>"learning weight".</em>
</ul>

 
<h3>How to use</h3>
<pre>
python mulitpleLinearRegression.py
</pre>
		
		
<h3>Expected Output</h3>
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

<h3>Requirements</h3>
 <p><a href="https://www.python.org/">Python (>2.7)</a>, <a href="http://www.numpy.org/">Numpy</a> and <a href="https://pandas.pydata.org/">Pandas</a>.</p>
 
 

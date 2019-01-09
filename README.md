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

 


<p align="center"><img src="https://rawgit.com/DrIanGregory/MachineLearning-MultiLinearRegressionWithGradientDescent (fetch/master/svgs/de06882459b39a41989eb4cfb3adad12.svg?invert_in_darkmode" align=middle width=100.40293725pt height=17.8466442pt/></p>
<u>Where:</u><br>
y is the target,<br>
<img src="https://rawgit.com/DrIanGregory/MachineLearning-MultiLinearRegressionWithGradientDescent (fetch/master/svgs/08a0aa2c6ce40306bad8bab7f60a9523.svg?invert_in_darkmode" align=middle width=18.32105549999999pt height=14.15524440000002pt/> is the intercept (bias value)<br>
W is a vector parameters (weights) <strong>to be estimated</strong>.

<p align="center"><img src="https://rawgit.com/DrIanGregory/MachineLearning-MultiLinearRegressionWithGradientDescent (fetch/master/svgs/e494bcd9ee6c4318551298c101e2fd8b.svg?invert_in_darkmode" align=middle width=85.88028735pt height=108.49422870000001pt/></p>

 
X is a matrix of 1's and K feature weights and N data points of <strong>given inputs</strong>

<p align="center"><img src="https://rawgit.com/DrIanGregory/MachineLearning-MultiLinearRegressionWithGradientDescent (fetch/master/svgs/7c0a4d8bcc24c33ddb67f85bf718d175.svg?invert_in_darkmode" align=middle width=200.3263218pt height=88.76800184999999pt/></p> 
 
 and <img src="https://rawgit.com/DrIanGregory/MachineLearning-MultiLinearRegressionWithGradientDescent (fetch/master/svgs/7ccca27b5ccc533a2dd72dc6fa28ed84.svg?invert_in_darkmode" align=middle width=6.672392099999992pt height=14.15524440000002pt/> is a vector of estimation errors denoted
 <p align="center"><img src="https://rawgit.com/DrIanGregory/MachineLearning-MultiLinearRegressionWithGradientDescent (fetch/master/svgs/8f0e92ede6c8f98716d5e718611b7c7b.svg?invert_in_darkmode" align=middle width=80.78418644999999pt height=88.76800184999999pt/></p>

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

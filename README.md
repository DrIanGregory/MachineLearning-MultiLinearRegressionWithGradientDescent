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


<p align="center"><img src="svgs/de06882459b39a41989eb4cfb3adad12.svg" align=middle width=100.40293725pt height=17.8466442pt/></p>
<u>Where:</u><br>
y is the target,<br>
<p><img src="svgs/08a0aa2c6ce40306bad8bab7f60a9523.svg" align=middle width=18.32105549999999pt height=14.15524440000002pt/> is the intercept (bias value)<p><br>
W is a vector parameters (weights) <strong>to be estimated</strong>.

<p align="center"><img src="svgs/e494bcd9ee6c4318551298c101e2fd8b.svg?invert_in_darkmode" align=middle width=85.88028735pt height=108.49422870000001pt/></p>

 
X is a matrix of 1's and K feature weights and N data points of <strong>given inputs</strong>

<p align="center"><img src="svgs/7c0a4d8bcc24c33ddb67f85bf718d175.svg?invert_in_darkmode" align=middle width=200.3263218pt height=88.76800184999999pt/></p> 
 
 and <img src="svgs/7ccca27b5ccc533a2dd72dc6fa28ed84.svg?invert_in_darkmode" align=middle width=6.672392099999992pt height=14.15524440000002pt/> is a vector of estimation errors denoted
 <p align="center"><img src="svgs/9720c5e6a6e5e815ac04a66c9acc4fc6.svg?invert_in_darkmode" align=middle width=69.6483381pt height=88.76800184999999pt/></p>


 
<ul style="list-style-type:disc">
	<li>The loss function chosen is <strong>minimum mean square error</strong> given by:</li>
</ul>

<p align="center"><img src="svgs/7fdf46eb804213abbe366918f7fb3ce7.svg?invert_in_darkmode" align=middle width=231.84320774999998pt height=47.60747145pt/></p>

<ul style="list-style-type:disc">
	<li>With partial derivatives</li>
</ul>
<p align="center"><img src="svgs/7e7d6721153a3f931a9c93332de39e07.svg?invert_in_darkmode" align=middle width=242.92115924999996pt height=105.07795814999999pt/></p>

<ul style="list-style-type:disc">
 <li>With weight updates given by:</li>
</ul>

<p align="center"><img src="svgs/7d5659f6aef43ad887de68ba61e98142.svg?invert_in_darkmode" align=middle width=126.34763954999998pt height=36.2778141pt/></p>

<ul style="list-style-type:disc">
	<li>Where <img src="svgs/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode" align=middle width=10.57650494999999pt height=14.15524440000002pt/> is the <em>"learning weight".</em>
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
 
 

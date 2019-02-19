# Import libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import seaborn

from random import random, seed
from matplotlib import cm

from mpl_toolkits.mplot3d import *
import matplotlib.pyplot as plt
import numpy as np
from random import random, seed
from matplotlib import cm
import scipy.interpolate as interp

import matplotlib;
# Used to write Latex on charts:
#matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['text.latex.unicode'] = True



def rmse(Y, Ypred):
    # Purpose: Root Mean Square Error (RMSE)
    # Useful in Model Evaluation.
    rmse = np.sqrt(sum((Y - Ypred) ** 2) / len(Y))
    return rmse

def r2Score(Y, Y_pred):
    # Purpose: Calculate R2 for goodness of fit.
    # Useful in Model Evaluation.
    mean_y = np.mean(Y)
    ss_tot = sum((Y - mean_y) ** 2)
    ss_res = sum((Y - Y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2









def updateActiveCost(i,x,y,plt,ax,numFrames,wHistory):

    currentDataPosition = int(len(x) / numFrames) * (i + 1);  # Adding a plus one to make the number a whole number so it looks nicer.

    if (i==0):
        newX = x[0:currentDataPosition]
        newY = y[0:currentDataPosition]
    else:
        previousDataPosition = int(len(x) / numFrames) * (i);  # Adding a plus one to make the number a whole number so it looks nicer.


        newX = x[previousDataPosition:currentDataPosition]
        newY = y[previousDataPosition:currentDataPosition]

    label = 'timestep {0}'.format(i) + " Iteration #: {0}".format(currentDataPosition)
    print(label)

    # if (i==0):
    #     oldX=x[0];
    #     oldY=y[0];
    #
    #     newX = x[0:currentDataPosition * (i + 1)]
    #     newY = y[0:currentDataPosition * (i + 1)]
    #
    # else:
    #     newX = x[currentDataPosition * (i):currentDataPosition * (i + 1)]
    #     newY = y[currentDataPosition * (i):currentDataPosition * (i + 1)]

        # oldX = x[currentDataPosition * (i-1):currentDataPosition * (i)]
        # oldY = y[currentDataPosition * (i-1):currentDataPosition * (i)]




    ax.scatter(newX, newY,s=0.2,c='blue',  marker='.', linewidths=None)

    #plt.plot(x[0], y[0], color='green', marker='.', markersize=10, linestyle='none')

    # if (i == 0):
    #     ax.scatter(oldX, oldY, c="blue", marker=',',s=0.2);  # The last point on the chart is coloured.
    #     ax.scatter(newX,newY,c="red", marker=',',s=0.2);   # The last point on the chart is coloured.
    # else:
    #     # The last point on the chart is coloured.
    #     ax.scatter(oldX[-1], oldY[-1], c="blue", marker=',',s=0.2)
    #     ax.scatter(newX[-1], newY[-1], c="red", marker=',',s=10);


    #ax.get_xticklabels()[3].set_color("red");  # Colour of a particular xlabel tick.


    # Update the line and the axes (with a new xlabel). Return a tuple of
    # "artists" that have to be redrawn for this frame.


    #plt.title("Multi Linear Regression Using Gradient Descent" + "\n" + "Iteration #: {0}".format(maxNumIterations) + "\n" + "y = {0} + {1}x + {2}y".format(round(newW[0],2), round(newW[1],2), round(newW[2],2), 4), fontdict=font)



    #labelEquation = ax.text(5000, 12.4, "job = {0} + {1}AI + {2}ML".format(round(wHistory[newX[-1]][0], 2),round(wHistory[newX[-1]][1], 2),round(wHistory[newX[-1]][2], 2)), fontsize=12)
    #labelIteraionNumber = ax.text(5000, 12.2, "Iteration Number: {0}".format(round(newX[-1], 0)), fontsize=12)

    ax.set_title("Iteration #: {0}".format(currentDataPosition) + "\n" + "job = {0} + {1}AI + {2}ML".format(round(wHistory[newX[-1]][0], 2),round(wHistory[newX[-1]][1], 2),round(wHistory[newX[-1]][2], 2)), fontsize=7)

    return plt, ax



def plottingActiveCost(costHistory,wHistory):

    fig = plt.figure("gradientDescentLinearRegressionConvergence")
    #fig.set_tight_layout(True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]);      # Takes into account suptitle is used and title doesn't overlap with it.
    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'bold',
            'size': 8,
            }

    plt.suptitle("Multi Linear Regression Using Gradient Descent", fontdict=font)

    plt.style.use('classic')
    ax = plt.gca()
    ax = plt.axes(facecolor ='#E6E6E6') # use a gray background.
    ax.set_axisbelow(True)
    plt.grid(color='w', linestyle='solid') # draw solid white grid lines.
    # hide axis spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    # hide top and right ticks
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    # lighten ticks and labels
    ax.tick_params(colors='gray', direction='out')
    for tick in ax.get_xticklabels():
        tick.set_color('gray')
    for tick in ax.get_yticklabels():
        tick.set_color('gray')

    ax.get_xticklabels()[0].set_fontweight("bold")
    ax.get_xticklabels()[0].set_color("darkred") ; # Colour of a particular xlabel tick.
    ax.scatter(0,costHistory[0],c="red");   # The last point on the chart is coloured.

    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')));      # Seaprate 000 with ,.
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'));                                    # 2dp for y axis.
    ax.xaxis.set_tick_params(labelsize=8); # Tick label size.
    ax.yaxis.set_tick_params(labelsize=8);  # Tick label size.
    x = list(range(len(costHistory)));
    y = costHistory;
    ax.set_xlim([min(x)-1000, max(x)])
    ax.set_ylim([min(y)-0.2, 12.65])

    plt.plot(x[0], y[0], color='green', marker='.', markersize=10, linestyle='none')
    plt.xlabel('Iterations (epochs)', fontsize=9)
    plt.ylabel('Cost', fontsize=9)
    plt.title('Rate of Convergence (Learning)');

    numFrames=10;


    anim = FuncAnimation(fig, updateActiveCost, fargs=(x,y,plt,ax,numFrames,wHistory),frames=np.arange(0, numFrames), interval=200)

    # NOTE: Under Windows 10, saving animation requires the FFmpeglibraries from: https://sourceforge.net/projects/imagemagick/.
    # It is likely FFmpeg is already installed in C:\Program Files\ffmpeg. But it doesn't seem to work. Best to use the Windows binary installer.
    anim.save('linearRegressionCost.gif', dpi=80, writer='imagemagick')

    #
    # # Plot a scatter that persists (isn't redrawn) and the initial line.
    # x = np.arange(0, 20, 0.1)
    # ax.scatter(x, x + np.random.normal(0, 3.0, len(x)))
    # line, = ax.plot(x, x - 5, 'r-', linewidth=2)
    #
    # anim = FuncAnimation(fig, update, fargs=(x,line,ax),frames=np.arange(0, 10), interval=200)
    #
    # # NOTE: Under Windows 10, saving animation requires the FFmpeglibraries from: https://sourceforge.net/projects/imagemagick/.
    # # It is likely FFmpeg is already installed in C:\Program Files\ffmpeg. But it doesn't seem to work. Best to use the Windows binary installer.
    # anim.save('C:/Projects/Coding/sandbox/pythonChartAnimation/line.gif', dpi=80, writer='imagemagick')


    plt.show()













def plottingStaticFit(X,Y,newW,maxNumIterations):

    # Create a grid covering the domain of the data:
    xMin = np.min(X[:, 1])
    xMax = np.max(X[:, 1])
    xStepSize=20*(xMax-xMin)/len(X[:, 1])

    yMin = np.min(X[:, 2])
    yMax = np.max(X[:, 2])
    yStepSize = 20*(yMax - yMin) / len(X[:, 1])

    xx, yy = np.meshgrid(np.arange(xMin, xMax, xStepSize), np.arange(yMin, yMax, yStepSize))

    # Evaluate the model on the grid.
    Z = newW[0] + newW[1]*xx + newW[2]*yy;

    # Plot scatter points and the fitted surface:
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot_surface(xx, yy, Z, rstride=1, cstride=1, alpha=0.3)
    x = X[:,1];
    y = X[:,2]
    z = Y;
    ax.scatter(x, y, z, c='red', s=3,marker='^');  # plot a 3d scatter plot.  s : marker size. c : colour.


    ax.axis('equal')
    ax.axis('tight')
    ax.invert_xaxis()
    ax.invert_yaxis()



    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'bold',
            'size': 10,
            }

    plt.suptitle("Multi Linear Regression Using Gradient Descent" + "\n" + "Iteration #: {0}".format(maxNumIterations) + "\n" + "y = {0} + {1}x + {2}y".format(round(newW[0],2), round(newW[1],2), round(newW[2],2), 4), fontdict=font)
    #plt.suptitle("Multi Linear Regression Using Gradient Descent", fontdict=font)

    #plt.text(2, 0.65, r'$\cos(2 \pi t) \exp(-t)$', fontdict=font)# Place the equation on the chart. First 2 inputs X,Y. Where 1,1 is top right. 0,0 bottom left.
    plt.xlabel('AI Skills')
    plt.ylabel('Machine Learning Skills')
    ax.set_zlabel('Job Prospects')

    ax.xaxis.label.set_size(8.5)
    ax.yaxis.label.set_size(8.5)
    ax.zaxis.label.set_size(8.5)

    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.tick_params(axis='both', which='minor', labelsize=7)


    # Retrieve the chart zoom and tilt settings (camera view). I use this to generate the right view for the plot so I can start an animation on it automatically.
    xlm = ax.get_xlim3d()  # These are two tupples
    ylm = ax.get_ylim3d()  # we use them in the next
    zlm = ax.get_zlim3d()  # graph to reproduce the magnification from mousing
    azm = ax.azim
    ele = ax.elev

    # Set the camera view (used when reproducing charts):
    azm = 137;
    ele = 23.18;
    ax.view_init(elev=ele, azim=azm)  # Reproduce view
    xlm = [56.358105503857445, 82.7858711162032];
    ylm = [56.923444812107185, 85.14853469323654];
    zlm = [7.07575499646555, 82.78294166317764];
    ax.set_xlim3d(xlm[0], xlm[1]);  # Reproduce magnification
    ax.set_ylim3d(ylm[0], ylm[1]);  # ...
    ax.set_zlim3d(zlm[0], zlm[1]);  # ...

    ax.set_xlim3d(min(X[:,1]), max(X[:,1]))
    ax.set_ylim3d(min(X[:,2]), max(X[:,2]))
    ax.set_zlim(min(Y), max(Y))


    plt.show()



def update(i,x,line,ax):
    label = 'timestep {0}'.format(i)
    print(label)
    # Update the line and the axes (with a new xlabel). Return a tuple of
    # "artists" that have to be redrawn for this frame.
    line.set_ydata(x - 5 + i)
    ax.set_xlabel(label)
    return line, ax


def plottingStaticCost(costHistory):


    fig = plt.figure("gradientDescentLinearRegressionConvergence")
    plt.suptitle("Multi Linear Regression Using Gradient Descent", fontdict=font)
    fig.set_tight_layout(True)

    plt.style.use('classic')
    ax = plt.gca()
    ax = plt.axes(facecolor ='#E6E6E6') # use a gray background.
    ax.set_axisbelow(True)
    plt.grid(color='w', linestyle='solid') # draw solid white grid lines.
    # hide axis spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    # hide top and right ticks
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    # lighten ticks and labels
    ax.tick_params(colors='gray', direction='out')
    for tick in ax.get_xticklabels():
        tick.set_color('gray')
    for tick in ax.get_yticklabels():
        tick.set_color('gray')

    ax.get_xticklabels()[3].set_color("red") ; # Colour of a particular tick.
    ax.scatter(len(costHistory),costHistory[-1],c="red");

    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')));      # Seaprate 000 with ,.
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'));                                    # 2dp for y axis.
    ax.xaxis.set_tick_params(labelsize=8); # Tick label size.
    ax.yaxis.set_tick_params(labelsize=8);  # Tick label size.
    x = list(range(len(costHistory)));
    y = costHistory;
    ax.set_xlim([min(x)-1000, max(x)])
    ax.set_ylim([min(y)-0.2, 12.65])

    plt.plot(x, y, color='blue', marker='.', markersize=0, linestyle='none')
    plt.xlabel('Iterations (epochs)', fontsize=9)
    plt.ylabel('Cost', fontsize=9)
    plt.title('Rate of Convergence (Learning)');







def costFunction(X, Y, W):
    # Purpose: Calculate the cost function for multi-linear regression.
    # Inputs:
    #       X       :  Data inputs.
    #       Y       :  Data inputs.
    #       W       :  Weights.
    N = len(Y)
    C = np.sum((X.dot(W) - Y) ** 2)/(2 * N);
    return C

def gradientDescent(X, Y, W, alpha, maxNumIterations=10000):
    # Purpose: Perform gradient decent algorithm given dataset.
    # Inputs:
    #       X                   :       X data      :       Pandas Dataframe of 1 or more columns..
    #       Y                   :       Y data.     :       Pandas Dataframe of 1 column.
    #       alpha               :       Optimisation learning rate.
    #       maxNumIterations    :       Optimisation maximum number of iterations.

    N = len(Y)
    costHistory=[];
    wHistory=[];
    iteration = 0;
    while iteration <maxNumIterations:
        # Hypothesis Values
        h = X.dot(W)
        # Difference b/w Hypothesis and Actual Y
        loss = h - Y
        # Gradient Calculation
        gradient = X.T.dot(loss) / N
        # Updating slope values using Gradient
        W = W - alpha * gradient
        # New Cost Value
        cost = costFunction(X, Y, W);
        costHistory.append(cost);     # Record the cost (not needed, nice to have for performance analysis).
        iteration = iteration+1;

        wHistory.append(W); # Record the weights at each iteration.
    return W, costHistory,wHistory




def programBody(data,alpha,maxNumIterations):

    # #-------------------------------------------------
    # # Plot the data in a scatter plot
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(math, read, write, color='#ef1234')
    # plt.show()
    # #-------------------------------------------------


    # Initialise variables:
        # Initial Coefficients:
    numXColumns = data.shape[1]-1;
    W = (numXColumns+1)*[0];         # Intercepts initialised to zero for the number of features supplied.
    x0 = np.ones(data.shape[0]);

    X = np.column_stack((x0,data.iloc[:, 1:(numXColumns + 1)].values)); # Supplied X data.
    Y = np.array(data.iloc[:,0]);            # Supplied Y data.

    newW, costHistory,wHistory = gradientDescent(X, Y, W, alpha,maxNumIterations)

    showResults(X, Y, W, newW, costHistory, maxNumIterations,wHistory);

    return


def showResults(X,Y,W,newW,costHistory,maxNumIterations,wHistory):
    # Purpose: To display the results.

    inital_cost = costFunction(X, Y, W);
    Y_pred = X.dot(newW)

    dash = '=' * 80;
    print(dash)
    print("MULTI LINEAR REGRESSION USING GRADIENT DESCENT TERMINATION RESULTS")
    print(dash)
    print("Initial Weights were:    {:>12.1f}, {:>2.1f}, {:>2.1f}.".format(W[0],W[1],W[2]))
    print("   With initial cost:    {:>12.1f}.".format(inital_cost))
    print("        # Iterations:    {:>12,.0f}.    ".format(maxNumIterations))
    print("       Final weights:    w0:{:>+0.2f}, w1:{:>+3.2f}, w2:{:>+3.3f}.".format(newW[0], newW[1], newW[2]))
    print("          Final cost:    {:>+12.1f}.".format(costHistory[-1]))
    print("                RMSE:    {:>+12.1f}, R-Squared: {:>+12.1f}".format(rmse(Y, Y_pred),r2Score(Y, Y_pred)))
    print(dash)



    #Charts:
        # Animated Charts:
    plottingActiveCost(costHistory, wHistory)
    plottingActiveFit(X,Y,newW,maxNumIterations,costHistory,wHistory)
        # Static Charts:
    plottingStaticFit(X,Y,newW,maxNumIterations)
    plottingStaticCost(costHistory)






def plottingActiveFit(X,Y,newW,maxNumIterations,costHistory,wHistory):
    def updateActiveFit(i, X, ax, numFrames, wHistory, newW, mySurfacePlot):
        label = 'timestep {0}'.format(i)
        print(label)

        currentDataPosition = int((i ) * len(wHistory) / numFrames);  # Adding a plus one to make the number a whole number so it looks nicer.

        # if (i==0):
        #
        #
        # else:
        #     newX = x[currentDataPosition * (i):currentDataPosition * (i + 1)]
        #     newY = y[currentDataPosition * (i):currentDataPosition * (i + 1)]
        #
        #     oldX = x[currentDataPosition * (i-1):currentDataPosition * (i)]
        #     oldY = y[currentDataPosition * (i-1):currentDataPosition * (i)]
        #

        # Create a grid covering the domain of the data:
        xMin = np.min(X[:, 1])
        xMax = np.max(X[:, 1])
        xStepSize = 20 * (xMax - xMin) / len(X[:, 1])

        yMin = np.min(X[:, 2])
        yMax = np.max(X[:, 2])
        yStepSize = 20 * (yMax - yMin) / len(X[:, 1])

        xx, yy = np.meshgrid(np.arange(xMin, xMax, xStepSize), np.arange(yMin, yMax, yStepSize))

        # Evaluate the model on the grid.
        Z = wHistory[currentDataPosition][0] + wHistory[currentDataPosition][1] * xx + wHistory[currentDataPosition][2] * yy;

        mySurfacePlot[0].remove()
        #mySurfacePlot[0]=ax.plot_surface(xx, yy, Z, rstride=1, cstride=1, alpha=0.3,cmap="magma")
        mySurfacePlot[0] = ax.plot_surface(xx, yy, Z, rstride=1, cstride=1, alpha=0.3,cmap="magma")

        fontBlack = {'family': 'serif',
                     'color': 'black',
                     'weight': 'bold',
                     'size': 8,
                     }

        # ax.set_title("Iteration #: {0}".format(currentDataPosition) + "\n" + "job = {0} + {1}AI + {2}ML".format(round(wHistory[currentDataPosition][0], 2),round(wHistory[currentDataPosition][1], 2),round(wHistory[currentDataPosition][2], 2)), fontsize=7)
        plt.suptitle("Iteration #: {:,}".format(currentDataPosition) + "\n" + "job = {0} + {1}AI + {2}ML".format(
            round(wHistory[currentDataPosition][0], 2), round(wHistory[currentDataPosition][1], 2),
            round(wHistory[currentDataPosition][2], 2)), fontdict=fontBlack)
        return plt, ax, mySurfacePlot




    # Plot scatter points and the fitted surface:
    fig = plt.figure("gradientDescentLinearRegressionConvergence")
#    plt.style.use('classic')
    ax = plt.gca(projection='3d')
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95]);  # Takes into account suptitle is used and title doesn't overlap with it.

    x = X[:,1];
    y = X[:,2]
    z = Y;
    ax.scatter(x, y, z, c='red', s=3,marker='^');  # plot a 3d scatter plot.  s : marker size. c : colour.

    ax.axis('equal')
    ax.axis('tight')
    ax.invert_xaxis()
    ax.invert_yaxis()



    # Create a grid covering the domain of the data:
    xMin = np.min(X[:, 1])
    xMax = np.max(X[:, 1])
    xStepSize=20*(xMax-xMin)/len(X[:, 1])

    yMin = np.min(X[:, 2])
    yMax = np.max(X[:, 2])
    yStepSize = 20*(yMax - yMin) / len(X[:, 1])

    xx, yy = np.meshgrid(np.arange(xMin, xMax, xStepSize), np.arange(yMin, yMax, yStepSize))

    # Evaluate the model on the grid.
    Z = wHistory[0][0] + wHistory[0][1]*xx + wHistory[0][2]*yy;

    mySurfacePlot = [ax.plot_surface(xx, yy,Z, color='0.75', rstride=1, cstride=1)]
    #mySurfacePlot = ax.plot_surface(xx, yy, Z, rstride=1, cstride=1, alpha=0.3)



    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'bold',
            'size': 8,
            }

    fontBlack = {'family': 'serif',
            'color': 'black',
            'weight': 'bold',
            'size': 8,
            }

    #plt.suptitle("Multi Linear Regression Using Gradient Descent" + "\n" + "Iteration #: {0}".format(maxNumIterations) + "\n" + "y = {0} + {1}x + {2}y".format(round(newW[0],2), round(newW[1],2), round(newW[2],2), 4), fontdict=font)
    plt.suptitle("Multi Linear Regression Using Gradient Descent", fontdict=font)
    plt.title("", fontdict=fontBlack)

    #plt.text(2, 0.65, r'$\cos(2 \pi t) \exp(-t)$', fontdict=font)# Place the equation on the chart. First 2 inputs X,Y. Where 1,1 is top right. 0,0 bottom left.
    plt.xlabel('AI Skills')
    plt.ylabel('Machine Learning Skills')
    ax.set_zlabel('Job Prospects')

    ax.xaxis.label.set_size(8.5)
    ax.yaxis.label.set_size(8.5)
    ax.zaxis.label.set_size(8.5)

    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.tick_params(axis='both', which='minor', labelsize=7)


    # Retrieve the chart zoom and tilt settings (camera view). I use this to generate the right view for the plot so I can start an animation on it automatically.
    xlm = ax.get_xlim3d()  # These are two tupples
    ylm = ax.get_ylim3d()  # we use them in the next
    zlm = ax.get_zlim3d()  # graph to reproduce the magnification from mousing
    azm = ax.azim
    ele = ax.elev

    # Set the camera view (used when reproducing charts):
    azm = 137;
    ele = 23.18;
    ax.view_init(elev=ele, azim=azm)  # Reproduce view
    xlm = [56.358105503857445, 82.7858711162032];
    ylm = [56.923444812107185, 85.14853469323654];
    zlm = [7.07575499646555, 82.78294166317764];
    ax.set_xlim3d(xlm[0], xlm[1]);  # Reproduce magnification
    ax.set_ylim3d(ylm[0], ylm[1]);  # ...
    ax.set_zlim3d(zlm[0], zlm[1]);  # ...

    ax.set_xlim3d(min(X[:,1]), max(X[:,1]))
    ax.set_ylim3d(min(X[:,2]), max(X[:,2]))
    ax.set_zlim(min(Y), max(Y))


    numFrames=10;
    anim = FuncAnimation(fig, updateActiveFit, fargs=(X,ax,numFrames,wHistory,newW,mySurfacePlot),frames=np.arange(0, numFrames), interval=200)
    anim.save('C:/Projects/Coding/sandbox/linearRegression/test/linearRegressionFit.gif', dpi=80, writer='imagemagick')



def run():
    # ----------------------------------------------------------------------------------------------------------------
    # User Inputs:
        # Optimisation Parameters :
    alpha = 0.0001;  # Optimisation learning rate.
    maxNumIterations = 2500000;  # Maximum number of optimisation iterations.

    # NOTE: Y labelled data is in column 1. All weight data (X) in following columns.
    #fileName = 'dataKaggleBoston.csv';  # File Name for data.
    fileName = 'student.csv';  # File Name for data.
    # ----------------------------------------------------------------------------------------------------------------


    # Generate Data:
    # Seed random number generator.
    np.random.seed(1234)
    numDataPoints=500;
    means = [70, 70]
    stds = [9,9]
    corr = 0.8  # correlation
    covs = [[stds[0] ** 2, stds[0] * stds[1] * corr],[stds[0] * stds[1] * corr, stds[1] ** 2]]
    data1 = np.random.multivariate_normal(means, covs, numDataPoints).T

    JobProbabilities = (data1[0] + data1[1])/2.5 + np.random.normal(loc=25, scale=4, size=numDataPoints);
    #AIScores = np.random.normal(loc=70, scale=15, size=1000);
    #MachineLearningScores = np.random.normal(loc=65, scale=15, size=1000);
    data = np.vstack((JobProbabilities, data1));
    data = pd.DataFrame({"JobPotential":data[0],"AI":data[1],"MachineLearning":data[2]});


    programBody(data, alpha, maxNumIterations);
    print("Finished");


if __name__ == '__main__':
    run()





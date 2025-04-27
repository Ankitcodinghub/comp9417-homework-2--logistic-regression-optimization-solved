# comp9417-homework-2--logistic-regression-optimization-solved
**TO GET THIS SOLUTION VISIT:** [COMP9417 Homework 2- Logistic Regression & Optimization Solved](https://www.ankitcodinghub.com/product/comp9417-machine-learning-solved-4/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;121355&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;COMP9417 Homework 2- Logistic Regression \u0026amp; Optimization Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
Homework 2: Logistic Regression &amp; Optimization

Introduction In this homework, we will first explore some aspects of Logistic Regression and performing inference for model parameters. We then turn our attention to gradient based optimisation, the workhorse of modern machine learning methods.

• Question 1 e): 1 mark

• Question 2 c): 1 mark

What to Submit

• A single PDF file which contains solutions to each question. For each question, provide your solution in the form of text and requested plots. For some questions you will be requested to provide screen shots of code used to generate your answer — only include these when they are explicitly asked for.

• .py file(s) containing all code you used for the project, which should be provided in a separate .zip file. This code must match the code provided in the report.

1

• You cannot submit a Jupyter notebook; this will receive a mark of zero. This does not stop you from developing your code in a notebook and then copying it into a .py file though, or using a tool such as nbconvert or similar.

• We will set up a Moodle forum for questions on this homework. Please read the existing questions before posting new questions. Please do some basic research online before posting questions. Please only post clarification questions. Any questions deemed to be fishing for answers will be ignored and/or deleted.

• Please check the Moodle forum for updates to this spec. It is your responsibility to check for announcements about the spec.

• Please complete your homework on your own, do not discuss your solution with other people in the course. General discussion of the problems is fine, but you must write out your own solution and acknowledge if you discussed any of the problems in your submission (including their name and zID).

When and Where to Submit

• Submission must be done through Moodle, no exceptions.

Question 1. Regularized Logistic Regression &amp; the Bootstrap

In this problem we will consider the dataset provided in Q1.csv, with binary response variable Y , and 45 continuous features X1,…,X45. Recall that Regularized Logistic Regression is a regression model used when the response variable is binary valued. Instead of using mean squared error loss as in standard regression problems, we instead minimize the log-loss, also referred to as the cross entropy loss. For a parameter vector β = (β1,…,βp) ∈ Rp, yi ∈ {0,1}, xi ∈ Rp for i = 1,…,n, the log-loss is

,

where s(z) = (1 + e−z)−1 is the logistic sigmoid (see Homework 0 for a refresher.) In practice, we will usually add a penalty term, and consider the optimisation:

(βˆ0,βˆ) = argmin{CL(β0,β) + penalty(β)} (1)

β0,β

where the penalty is usually not applied to the bias term β0, and C is a hyper-parameter. For example, in the `1 regularisation case, we take penalty(β) = kβk1 (a LASSO for logistic regression).

(a) Consider the sklearn logistic regression implementation (section 1.1.11), which claims to minimize the following objective:

w,ˆ cˆ= argmin . (2)

It turns out that this objective is identical to our objective above, but only after re-coding the binary variables to be in {−1,1} instead of binary values {0,1}. That is, yei ∈ {−1,1}, whereas yi ∈ {0,1}. Argue rigorously that the two objectives (1) and (2) are identical, in that they give us the same solutions (βˆ0 = cˆand βˆ = wˆ). Further, describe the role of C in the objectives, how does it compare to the standard LASSO parameter λ? What to submit: some commentary/your working.

(b) Take the first 500 observations to be your training set, and the rest as the test set. In this part, we will perform cross validation over the choice of C from scratch (Do not use existing cross validation implementations here, doing so will result in a mark of zero.)

Create a grid of 100 C values ranging from C = 0.0001 to C = 0.6 in equally sized increments, inclusive. For each value of C in your grid, perform 10-fold cross validation (i.e. split the data into 10 folds, fit logistic regression (using the LogisticRegression class in sklearn) with the choice of C on 9 of those folds, and record the log-loss on the 10th, repeating the process 10 times.) For this question, we will take the first fold to be the first 50 rows of the training data, the second fold to be the next 50 rows, etc. Be sure to use `1 regularisation, and the liblinear solver when fitting your models.

To display the results, we will produce a plot: the x-axis should reflect the choice of C values, and for each C, plot a box-plot over the 10 CV scores. Report the value of C that gives you the best CV performance. Re-fit the model with this chosen C, and report both train and test accuracy using this model. Note that we do not need to use the ye coding here (the sklearn implementation is able to handle different coding schemes automatically) so no transformations are needed before applying logistic regression to the provided data. What to submit: a single plot, train and test accuracy of your final model, a screen shot of your code for this section, a copy of your python code in solutions.py

(c) In this part we will compare our results in the previous section to the sklearn implementation of gridsearch, namely, the GridSearchCV class. My initial code for this section looked like:

grid_lr = GridSearchCV(estimator=

LogisticRegression(penalty=’l1’, solver=’liblinear’),

cv=10, param_grid=param_grid)

grid_lr.fit(Xtrain, Ytrain)

1

2

3

4

5

6

7

We next explore the idea of inference. To motivate the difference between prediction and inference, see some of the answers to this stats.stachexchange post. Needless to say, inference is a much more difficult problem than prediction in general. In the next parts, we will study some ways of quantifying the uncertainty in our estimates of the logistic regression parameters. Assume for the remainder of this question that C = 1, and work only with the training data set (n = 500 observations) constructed earlier.

(d) In this part, we will consider the nonparametric bootstrap for building confidence intervals for each of the parameters β1,…,βp. (Do not use existing Bootstrap implementations here, doing so will result in a mark of zero.) To describe this method, let’s first focus on the case of βˆ1. The idea behind the nonparametric bootstrap is as follows:

1. Generate B bootstrap samples from the original dataset. Each bootstrap sample consists of n points sampled with replacement from the original dataset, where n is the size of the original dataset.

2. On each of the B bootstrap samples, compute an estimate of β1, giving us a total of B estimates which we denote .

3. Define the bootstrap mean and standard error respectively:

.

4. A 90% bootstrap confidence interval for β1 is then given by the interval:

((β˜1)L,(β˜1)U) = (5th quantile of the bootstrap estimates, 95th quantile of the bootstrap estimates)

The idea behind a 90% confidence interval is that it gives us a range of values for which we believe with 90% probability the true parameter lives in that interval. If the computed 90% interval contains the value of zero, then this provides us evidence that β1 = 0, which means that the first feature should not be included in our model.

Take B = 10000 and set a random seed of 12 (i.e. np.random.seed(12)). Generate a plot where the x-axis represents the different parameters β1,…,βp, and plot a vertical bar that runs from

(β˜p)L to (β˜p)U. For those intervals that contain 0, draw the bar in red, otherwise draw it in blue. Also indicate on each bar the bootstrap mean. Remember to use C = 1.0.

What to submit: a single plot, a screen shot of your code for this section, a copy of your python code in solutions.py

(e) Comment on your results in the previous section, what do the confidence intervals tell you about the underlying data generating distribution? How does this relate to the choice of C when running regularized logistic regression on this data? Is regularization necessary?

Question 2. Gradient Based Optimization

In this question we will explore some algorithms for gradient based optimization. These algorithms have been crucial to the development of machine learning in the last few decades. The most famous example is the backpropagation algorithm used in deep learning, which is in fact just an application of a simple algorithm known as (stochastic) gradient descent. The general framework for a gradient method for finding a minimizer of a function f : Rn → R is defined by

x(k+1) = x(k) − αk∇f(xk), k = 0,1,2,…, (3)

where αk &gt; 0 is known as the step size, or learning rate. Consider the following simple example of√

minimizing g(x) = 2 x3 + 1. We first note that g0(x) = 3×2(x3 + 1)−1/2. We then need to choose a starting value of x, say x(0) = 1. Let’s also take the step size to be constant, αk = α = 0.1. Then we have the following iterations:

x(1) = x(0) − 0.1 × 3(x(0))2((x(0))3 + 1)−1/2 = 0.7878679656440357 x(2) = x(1) − 0.1 × 3(x(1))2((x(1))3 + 1)−1/2 = 0.6352617090300827

x(3)= 0.5272505146487477

…

and this continues until we terminate the algorithm (as a quick exercise for your own benefit, code this up and compare it to the true minimum of the function which is x∗ = −1). This idea works for functions that have vector valued inputs, which is often the case in machine learning. For example, when we minimize a loss function we do so with respect to a weight vector, β. When we take the stepsize to be constant at each iteration, this algorithm is called gradient descent. For the entirety of this question, do not use any existing implementations of gradient methods, doing so will result in an automatic mark of zero for the entire question. (a) Consider the following optimisation problem:

min f(x), x∈Rn

where

,

and where A ∈ Rm×n, b ∈ Rm are defined as

.

Run gradient descent on f using a step size of α = 0.1 and starting point of x(0) = (1,1,1,1). You will need to terminate the algorithm when the following condition is met: k∇f(x(k))k2 &lt; 0.001. In your answer, clearly write down the version of the gradient steps (3) for this problem. Also, print out the first 5 and last 5 values of x(k), clearly indicating the value of k, in the form:

k = 0, x(k) = [1,1,1,1] k = 1, x(k) = ··· k = 2, x(k) = ···

…

What to submit: an equation outlining the explicit gradient update, a print out of the first 5 and last 5 rows of your iterations, a screen shot of any code used for this section and a copy of your python code in solutions.py.

(b) Note that using a constant step-size is sub-optimal. Ideally we would ideally like to take large steps at the beginning (when we are far away from the optimum), then take smaller steps as we move closer towards the minimum. There are many proposals in the literature for how best to choose the step size, here we will explore just one of them called the method of steepest descent. This is almost identical to gradient descent, except at each iteration k, we choose

αk = argminf(x(k) − α∇f(x(k))).

α≥0

In words, the step size is chosen to minimize an objective at each iteration of the gradient method, the objective is different at each step since it depends on the current x-value. In this part, we will run steepest descent to find the minimizer in (a). First, derive an explicit solution for αk (mathematically, please show your working). Then run steepest descent with the same x(0) as in (a), and α0 = 0.1. Use the same termination condition. Provide the first and last 5 values of x(k), as well as a plot of αk over all iterations. What to submit: a derivation of αk, a print out of the first 5 and last 5 rows of your iterations, a single plot, a screen shot of any code used for this section, a copy of your python code in solutions.py.

(c) Comment on the differences you observed, why would we prefer steepest descent over gradient descent? Why would you prefer gradient descent over steepest descent? Finally, explain why this is a reasonable condition to terminate use to terminate the algorithm.

In the next few parts, we will use the gradient methods explored above to solve a real machine learning problem. Consider the data provided in Q2.csv. It contains 414 real estate records, each of which contains the following features:

• age: age of property

• nearestMRT: distance of property to nearest supermarket

• nConvenience: number of convenience stores in nearby locations

• latitude

• longitude

The target variable is the property price. The goal is to learn to predict property prices as a function of a subset of the above features.

(d) We need to preprocess the data. First remove any rows with missing values. Then, delete all features except for age, nearestMRT and nConvenience. Then use the sklearn minmaxscaler to normalize the features. Finally, create a training set from the first half of the resulting dataset, and a test set from the remaining half. Your end result should look like:

• first row X train: [0.73059361,0.00951267,1.]

• last row X train: [0.87899543,0.09926012,0.3]

• first row X test: [0.26255708,0.20677973,0.1]

• last row X test: [0.14840183,0.0103754,0.9]

• first row Y train: 37.9

• last row Y train: 34.2

• first row Y test: 26.2

• last row Y test: 63.9

What to submit: a copy of your python code in solutions.py

(e) Consider the loss function

,

and consider the linear model

yˆi = w0 + w1xi1 + w2xi2 + w3xi3, i = 1,…,n.

We can write this more succinctly by letting w = (w0,w1,w2,w3)T and xi = (1,xi1,xi2,xi3)T, so that yˆi = wTxi. The mean loss achieved by our model (w) on a given dataset of n observations is then

.

We will run gradient descent to compute the optimal weight vector w. The iterations will look like

w(k+1) = w(k) − αk∇L(w).

Instead of computing the gradient directly though, we will rely on an automatic differentiation library called JAX. Read the first section of the documentation to get an idea of the syntax. Implement gradient descent from scratch and use the JAX library to compute the gradient of the loss function at each step. You will only need the following import statements:

# pip install –upgrade pip

# pip install jax

# pip install –upgrade jax[cpu]

import jax.numpy as jnp from jax import grad

1

2

3

4

5

6

7

8

Use w(0) = [1,1,1,1]T, and a step size of 1. Terminate your algorithm when the absolute value of the loss from one iteration to the other is less than 0.0001. Report the number of iterations taken, and the final weight vector. Further, report the train and test losses achieved by your final model, and produce a plot of the training loss at each step of the algorithm. What to submit: a single plot, the final weight vector, the train and test loss of your final model, a screen shot of your code for this section, a copy of your python code in solutions.py

(f) Finally, re-do the previous section but with steepest descent instead. In order to compute αk at each step, you can either use JAX or it might be easier to use the minimize function in scipy (See lab3). Run the algorithm with the same w(0) as above, and take α0 = 1 as your initial guess when numerically solving for αk (for each k). Terminate the algorithm when the loss value falls bellow 2.5. Report the number of iterations it took, as well as the final weight vector, and the train and test losses achieved. Generate a plot of the losses as before and include it. What to submit: a single plot, the final weight vector, the train and test accuracy of your final model, a screen shot of your code for this section, a copy of your python code in solutions.py

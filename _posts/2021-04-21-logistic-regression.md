---
title:  "The Complete(ish) Guide on Logistic Regression (with Math) for Dummies"
mathjax: true
layout: post
categories: media
---

So, let's talk about Logistic Regression. It is arguably the most common technique in machine learning for predicting two classes, and on the surface, it is relatively easy to understand and evaluate.

However, if you delve more deeply into the literature and math behind logistic regression, it can start to get a little bit confusing - you know, because of some of the math. But like everything, if you can understand the basic path to it and then build, things will becomes super easy. This is my attempt to make sense of everything about logistic regression. 

I am going to start from ground zero and build up with some examples in python along the way. We will do the whole algorithm from scratch!

## The Base - What do we use Logistic Regression for?

If you are first understanding logistic regression, you have to know one thing. You have to predict one of two classes, like cats vs dogs, fire vs. ice, etc. 
In data science, we typically give what we are trying to predict a name. We will call them labels, or Y. We have to codify the labels to be some number. 
Again, we typically give our Y the values 1 or 0, 1 corresponding to one label, and 0 for the other label.

<div style="text-align:center"><img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+Y+%3D%0A%5Cbegin%7Bcases%7D%0A1+%5C%5C%0A0%0A%5Cend%7Bcases%7D" 
alt="Y =
\begin{cases}
1 \\
0
\end{cases}"></div>

So we have a bunch of labels $$Y$$. How do we predict one of those classes? Well, you gotta have some data that represents each class. We typically call these features, or $$X$$.

Features represent the labels. For classifying cats and dogs, we may have one feature called color with many different color values (red, yellow, orange). 
We might have another feature called length, with varying lengths of the animal (20 inches, 50 inches, etc.). 
As you gather more features describing cats and dogs, you are developing a matrix that will have a bunch of columns (features) and rows (the values for one animal). 

Let's move to another more complicated example. I have a dataset downloaded from kaggle called the [South Africa Coronary Heart Disease Dataset](https://www.kaggle.com/emilianito/saheart) (SAHeart). We are going to be predicting whether or not a sample of males from South Africa have coronary heart disease based on a host of measured features, including systolic blood pressure, tobacco use, LDL cholesterol, adiposity, family history, type-A behavior, obesity, alcohol use, and age.

Each individual feature can be denoted as $$x_i$$. For example, $$x_1$$ represent the data in the "adiposity" feature in the SA Heart Dataset. 

We have about 462 rows, which means we have measured 462 patients. With this data, you could "train" an algorithm to predict if the person has coronary heart disease, 1 for "yes they have it", and 0 for "no they do not have it."

## How do we predict a class?

So you're probably wondering how do we get from a set of features, $$X$$, to a set of predictions, $$Y$$? We have to do a little bit of math. We need some equation to be able to take the features $$X$$ and get an output of 1 or 0. This equation will basically be "fitting" the numerical features to the numerical outcome (1 or 0). 
That means when a new set of features comes around, we can just use the fit equation to predict 1 or 0. Let's take a look at how this all works.

We need to define some notation first. Let's define the probability that a given set of features is 1.

$$ P(Y=1) = p $$

Now let's define the probability that a given set of features is 0.

$$ P(Y=0)= 1 - p $$

When we are doing logistic regression, we are predicting a probability, between 1 and 0. We will be using p for when the predicted class is 1, and 1 - p when the predicted class is 0.

So now let's take a look at something interesting. We can use $$p$$ and $$1-p$$ to get the odds that a class is 1 or 0 with this equation here.

$$\frac{p} {1-p} = \frac{P(Y=1)} {P(Y=0)}$$

This is known as the odds, which is $$p$$ divided by $$1-p$$. When the odds are 1, you have equal probability of getting a class. When odds are greater than 1, you will predict the class is 1. If less than 1, you will predict the class is 0. This is all well and good for understanding the relationships between probabilities, but it still doesn't help us get from features to predictions. We need something more.

Here comes the binary [logit function](https://en.wikipedia.org/wiki/Logit). This was formulated some time ago and has been critical in linking the probability values 1 and 0 to our features X. Odds are calculated above and below 1. However, if you take the log of the odds, you get the log odds, which maps values between 0 and 1. Logistic regression suggests a linear relationship between the features, $$X$$, and the log odds of the the event that $$Y=1$$,which is $$p$$. 

You can map the total relationship between the log odds and your features below.

$$logit(p) = \log{\left(\frac{p} {1-p}\right)} = log(p) - log(1-p) = -log\left(\frac{1} {p} - 1\right)$$

$$-log\left(\frac{1} {p} - 1\right) = \beta_{0} + \beta_{1}x_{1} + \dots + \beta_{p-1}x_{p-1} = \beta^TX$$


, where 
- $$p$$ is the probability that $$y=1$$,
- $$\beta_{1}$$ to $$\beta_{p-1}$$ are the weights/coefficients mapped to your features (each beta is one value) 
- $$x_{1}$$ to $$x_{p-1}$$ are each individual feature vector (vector (many values) for adiposity, etc.)
- $$\beta_{0}$$ is an intercept.

$$\beta^TX$$ are the weights and the features condensed into matrix form. You can do some cool stuff with this property. I will show that later in later equations.

So maybe you are starting to understand how we map outputs to features. It's all in the Betas! Those are the weights or coefficients "attached" to your numerical features. The logistic equation is trying to adjust those weights in the best possible based on all the data for a particular feature to give the best output for Y.

Now that we know this relationship between the features and the probabilities, how do we actually get the probability that y=1? We have to take that logit equation and exponentiate it. This will give us everything we need to make a prediction!

Step 1: Exponentiate the log odds to get the odds, but now we have the features exponentiated as well.

$$\frac{p} {1-p} = e^{ \beta_{0} + \beta_{1}x_{1} + \dots + \beta_{p-1}x_{p-1} }$$

Step 2: Solve for p

$$p = \frac{ e^{ \beta_{0} + \beta_{1}x_{1} + \dots + \beta_{p-1}x_{p-1}  } } {1 + e^{ \beta_{0} + \beta_{1}x_{1} + \dots + \beta_{p-1}x_{p-1}  }} = \frac{ e^{\beta^TX} } {1 + e^{\beta^TX}}$$

Step 3: Simplify

$$p = \frac{ 1 } {1 + e^{-\beta_{0} + \beta_{1}x_{1} + \dots + \beta_{p-1}x_{p-1}  }} = \frac{ 1 } {1 + e^{-\beta^TX }}$$

That's it! If you know the Betas and you have your feature values, all you have to do is plug and chug to get a prediction for the probability that $$y=1$$ ($$p$$). This final equation is called the **sigmoid function**, where the matrix multiplication of your Betas and your features $$X$$ is typically "$$z$$" in a lot of things you will read online. This will output 1 probability value pertaining to $$p$$. If you want to output 2 probabilities, one pertaining to $$p$$ and another pertaining to $$1-p$$, you would make predictions with what is called a [**softmax function**](https://en.wikipedia.org/wiki/Softmax_function). Softmax is typically used when predicting more than 2 classes, so we are just going to look at the sigmoid.

Let's see what this looks like in python code. 

{% highlight python %}
def logistic_regression_predict(W, X):
    '''
    Definition:
        return a matrix of probabilities from sigmoid function
    Inputs:
        W (numpy array): your weights array - shape (10,) - 10 features
        X (numpy array): your feature values matrix - shape (N, 10) - 10 features, N data
    Returns:
        probabilities (numpy array): probability matrix representing the probability a 
                                     data point will be in class 1
    '''
    probabilities = np.array(1.0 / (1.0 + np.exp(np.dot(X,-W))))
    return probabilities
{% endhighlight %}

Let's do an example with the AS dataset. I am going to just select some random weights. As you can see there is 1 weight/coefficient for each feature.

{% highlight python %}
# 3 Initialized Weights
>>> weights = np.array([0.0003, 0.433, -0.00045])
# We are looking at 4 data points and 3 Features
>>> features = np.array(X[['sbp', 'tobacco', 'ldl']])[:4] 
# predict p based on weights and feature values
>>> logistic_regression_predict(weights, features)
array([0.99473514, 0.51138441, 0.51711181, 0.96427809])
{% endhighlight %}

That's awesome! We can use features and beta values to calculate the probability of a class. But how do we get the best weights that fit the data in the most accurate way?
We will have to find a way of evaluating those weights, and then optimize those weights

## How do we evaluate our weights/parameters?

How do we get the best weights that fit the data in the most accurate way? We have to do what is called in the literature as maximum likelihood estimation!

How do you do all this? In the most simple terms, you need to find the points in your paramater space (the weights) that best minimize the error in your predictions from the actuals.
This must be done by just picking weights and seeing what happens. Then you update them and repeat until you get the best fit. When you get the best weights you can, you stick with them. 
This is what all the model training is. Let's get into it more technically. 

Maximum Likelihood Estimation in mathematical terms is a "[method of estimating the parameters of a probability distribution by maximizing a likelihood function, so that under the assumed statistical model the observed data is most probable](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation#:~:text=In%20statistics%2C%20maximum%20likelihood%20estimation,observed%20data%20is%20most%20probable.)". What is the likelihood function you maximize in logistic regression? There are actually quite a few ways to do the same thing. I will go into all 3 and say which one we will use for this example.

Option 1:
You can maximize the likelihood of your data with an equation called [log-likelihood](https://en.wikipedia.org/wiki/Likelihood_function#:~:text=Log%2Dlikelihood%20function%20is%20a,to%20maximizing%20the%20log%2Dlikelihood.), which is a log transform of the likelihood function.

Option 2:
You can take the inverse of the log-likelihood function. This is called a **loss function** and is specifically termed **binary cross entropy loss** or **log loss**. You will see both used interchangeably. The output of the loss function is a loss value. It tells you how far off your predicted probability is from the actual value. Your loss increases the further away your prediction (number between 1 and 0) is from the actual value (1 or 0).

Option 3:
You can do calculate categorical cross-entropy loss. This is usually done when there are multiple classes, and has close ties with the softmax function. I found this [blog](https://gombru.github.io/2018/05/23/cross_entropy_loss/) extremely useful in parsing these losses apart.

As you might have guessed, we are going with option 2, the log loss function. We will be evaluating weights (parameters) by looking at the output of the difference between actual and predicted with log loss. The goal of this log loss fucntion is each time we make a change to our weights (parameters), we want to see that the value from the log loss function goes down/gets minimized. When we have reached best minimization (convergence) on the loss, then we stick with those weights. Here is our log loss function:


$$ Logloss(y, p) = -((y) log(p) + (1-y)log(1-p)) $$

However, this log loss is not enough. This is a calculation on one prediction. Evaluating each prediction and then changing the weights each time will get tedious, especially as we are trianing on larger datasets. We want to look at our loss over many predictions, preferrably in batches. That is why we can also define a **cost function**. Here it is below. It is just the average of your losses on that batch of data.

$$ Cost = \frac {1} {m} \sum_{i=1}^{m} -((y) log(p) + (1-y)log(1-p)) $$

$$ Cost = \frac {1} {m} \sum_{i=1}^{m} logloss(y, p) $$

So if you have to remember anything - loss function is for one data point, cost function is for many data points.

Let's see how we can do evaluation with these functions in python!

{% highlight python %}
>>> test_y = 1 # our label
>>> p = 0.8    # our predicted probability
>>> loss = -( (test_y*np.log(p)) + ((1-test_y)*np.log(1-p)) )
>>> print(loss)
0.2231435513142097
{% endhighlight %}

As you can see above, when we plug the label and the probability in our loss function, we get a loss value. Let me show you something interesting.

{% highlight python %}
>>> test_y = 1 # our label
>>> p = 0.8    # our predicted probability
>>> loss = -( (test_y*np.log(p)) )
>>> print(loss)
0.2231435513142097
{% endhighlight %}

Same value! Why is that? If your class is y = 1, then the second side of the equation zeros out. If your class is y = 0, then the first side zeros out. Let's show when y is 0, but the predicted probability is a little off.

{% highlight python %}
>>> test_y = 0
>>> p = 0.8
>>> loss1 = -( (test_y*np.log(p)) + ((1-test_y)*np.log(1-p)) )
>>> print(loss1)
1.6094379124341005

>>> loss2 = -( ((1-test_y)*np.log(1-p)) )
>>> print(loss2)
1.6094379124341005
{% endhighlight %}

Did you notice that the output of your loss function increased as a result of a worse prediction? You have a higher penalty on classes that are misidentified. This becomes important during optimization steps, which we will discuss in the next section.

Here are the loss function and cost functions here:

{% highlight python %}
def loss_fn(y, p):
    '''
    Definition: get the log loss or the log difference between the actual label (y) and what is prediction (p)
    '''
    return -( (y*np.log(p)) + ((1-y)*np.log(1-p)) )

def cost_function(Y, P):
    '''
    Defintion: get the average log loss among all the training examples
    '''
    m = len(P)
    return 1/m*np.sum(loss_fn(Y, P))
{% endhighlight %}

Now if you wanted to implement this evaluation on a few examples, you would do the following:

{% highlight python %}
>>> p = logistic_regression_predict(weights, features)
>>> print(p)
[0.99473514 0.51138441 0.51711181 0.96427809]

>>> loss_fn(np.array(y).reshape(-1)[:4], p)
array([0.00527877, 0.67063371, 0.72797015, 0.03637555])

>>> cost_function(np.array(y).reshape(-1)[:4], p)
0.36006454524061055

{% endhighlight %}

## How do we get the best weights/parameters with those loss functions? Optimization Algorithms!

Now that you know what **maximium likelihood estimation** is and how to evaluate your predictions based on your parameters (weights) using a cost function, we can finally start finding the best weights!

Now we do the optimization using an optimization algorithm. Remember, the goal is to find a set of weights, W, which minimizes that cost function, averaged over all of our training examples.

**NOTE:** That cost function for logistic regression is **convex**. What this means is that function has just one minimum loss and there is no local minima go get stuck in. In other words, we can find a sweet spot for all of our weights where the loss is at the minimum it can be. When this happens, we call it **convergence**.

What optimizer should we use? There are a ton of them. Interestingly enough, the easiest one to use is gradient descent, which is the most common way to update the weights on a neural network! The scikit-learn Logistic Regression actually uses some optimizers similar to this - ‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’.  Those are other algorithms that are a little more refined optimizers than gradient descent with built in regularization, but the idea is still the same. 

How does gradient descent work with respect to logistic regression? Gradient Descent tells us whether we should make the weights bigger or smaller (move to the left or right) to reach the minimum loss. More specifically, it tells us which direction for the weights to go by finding the gradient of the loss function and at the current point and moving in the opposite direction.  Let's see some definitions below:

**Gradient** - a vector pointing in the direction of the greatest increase in a function. The gradient is a "multivariable" generalization of the slope of a line - If the slope is negative, we move our weight, $$w$$, in the opposite direction. In simple terms, the gradient is just a vector.

**Learning rate** - the magnitude of the amount to move the weight is the value of the slope multiplied by the learning rate. A higher learning rate moves $$w$$ more on each iteration.

We caclulate the gradient for that feature/variable, then update the new weight by subtracting our current weight from that gradient times the learning rate, $$\eta$$. If we were only updating one weight for one training example, it would look like this:

$$gradient = \frac{d} {dw}f(x;w)$$

$$w^{t+1} = w^t - \eta \frac{d} {dw}f(x;w)$$

Now if we have multiple weights and a bias term but one training example we make the terms we are trying to reach \theta. Here is the equation for that:

$$\theta^{t+1} = \theta^t - \eta \triangledown_\theta(logloss(y^i, p^i)))$$

The the $$\triangledown_\theta$$ is the gradient. Again, the $$\eta$$ is the learning rate. 

Now if we wanted to update more than one training example at a time, we add a little more flavor to our equation. It is now the average of the individual gradients. This will dramatically decrease training time:

$$ gradient = \frac{\partial Cost}{\partial w_j} $$

Now the new update is the following, which is what we will be implementing in python code.

$$\theta^{t+1} = \theta^t - \eta \triangledown_\theta Cost$$

Let's take a look at the gradient descent algorithm in python.

{% highlight python %}
def gradient_descent(features, labels, weights, learning_rate):
    '''
    Definition: batch gradient descent - compute the gradient over the whole set of training instances
    '''
    N = len(features)
    predictions_before_gd = logistic_regression_predict(weights, features)
    
    ## compute the dot product of the feature values * (predictions-labels) - gets the slopes for each feature 
    ## remember that the predictions is essentially the dot product of weights*features and returns p
    gradient        = np.dot(features.T, predictions_before_gd - labels) 
    updated_weights = weights - learning_rate*(gradient/N)

    predictions_after_gd = logistic_regression_predict(updated_weights, features)
    
    return updated_weights, predictions_after_gd
{% endhighlight %}

## Putting it all together to train logistic regression

That's great! We can make predictions. We can evaluate. We can update. Now we put all of these functions together to train our logistic regression model! We can set the number of iterations we want to go through in the training step. I will also implement an early stopping mechanism when the cost function stops changing very much after a certain number of iterations (convergence). Let's see what it looks like!


{% highlight python %}
def train(features, labels, weights, lr, iterations):
    cost_history = []
    convergence_number = 0

    for i in range(iterations):
        if convergence_number > 8:
            break

        # make predictions with gradient descent
        weights, predictions = gradient_descent(features, labels, weights, lr)
        # calculate cost
        cost = cost_function(labels, predictions)
        
        # early stopping mechanism
        if len(cost_history)>0:
            if np.abs(cost_history[-1] - cost) < 0.1:
                cost_history.append(cost)
                convergence_number+=1
            else:
                convergence_number=0
        cost_history.append(cost)

        # Log Training Progress
        if i % 1000 == 0:
            print("iter: "+str(i) + " cost: "+str(cost))

    return weights, cost_history

def classify(predictions):
    '''
    input  - N element array of predictions between 0 and 1
    output - N element array of 0s (False) and 1s (True)
    '''
    classifications = [] 
    for prediction in predictions:
        classifications.append(1 if prediction >= .5 else 0)
    return classifications

def accuracy(predicted_labels, actual_labels):
    diff = predicted_labels - actual_labels
    
    ## count how many different labels there and divide by the total number of labels
    return 1.0 - (float(np.count_nonzero(diff)) / len(actual_labels))
{% endhighlight %}

Let's implement on the SA Heart example!

{% highlight python %}
## Initialize the weights first
>>> initialized_weights = np.zeros(shape=np.array(X).shape[1])

## Train your algorithm
>>> weights, cost_history = train(np.array(X), np.array(y).reshape(-1), initialized_weights, 0.001, 100000)

## Get the final predictions
>>> p = logistic_regression_predict(weights, np.array(X))
{% endhighlight %}

Now we want to get the accuracy on our model. There are other calculations we can do with recall, precision, F-score. These are more valuable. I can devote another blog to that. For now, let's just look at accuracy.

{% highlight python %}
>>> predicted_labels = classify(p)
>>> accuracy(predicted_labels, np.array(y).reshape(-1))
0.6731601731601732
{% endhighlight %}

That's not amazing, but it's not bad! At least we got it to work, right? Let's see how a base logistic regression works in an scikit-learn backed model. I am using the [quikml](https://pypi.org/project/quikml/) library I created for evaluating binary classifiers. We will use the liblinear solver from scikit-learn

{% highlight python %}
>>> from quikml.models import LR
>>> lr = LR()
>>> params={'solver' : 'liblinear'}
>>> lr.train(np.array(X), np.array(y).ravel(), params=params)
0.734
{% endhighlight %}

How about that? Our made from scratch gradient descent algorithm wasn't that far off from a basic implementation of a native logistic regression model which uses a liblinear optimizer.

## Further Reading - TBC!

I encourage you to look up (regularization techniques)[https://stats.stackexchange.com/questions/228763/regularization-methods-for-logistic-regression/228785#228785]. These are important for picking out features. It utilizes the cost function to penalize features that are not predicting the probabilities very well. Lasso and Ridge regression are the most common techniques.

### Sources:
- https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html
- https://stats.stackexchange.com/questions/198038/cross-entropy-or-log-likelihood-in-output-layer
- https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#:~:text=Cross%2Dentropy%20loss%2C%20or%20log,So%20predicting%20a%20probability%20of%20.
- https://stats.stackexchange.com/questions/228763/regularization-methods-for-logistic-regression/228785#228785
- https://stats.stackexchange.com/questions/179026/objective-function-cost-function-loss-function-are-they-the-same-thing
- https://web.stanford.edu/~jurafsky/slp3/5.pdf
- https://datacadamia.com/data_mining/loss_function
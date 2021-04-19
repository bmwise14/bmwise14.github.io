---
title:  "Logistic Regression (with Math) for Dummies"
mathjax: true
layout: post
categories: media
use_math: true
---

So, let's talk about Logistic Regression. It is arguably the most common technique in machine learning for predicting two classes, and on the surface, it is relatively easy to understand and evaluate.

However, if you delve more deeply into the literature and math behind logistic regression, it can start to get a little bit confusing - you know, because of some of the math. But like everything, if you can understand the basic path to it and then build, things will becomes super easy. This is my attempt to make sense of everything about logistic regression. 

I am going to start from ground zero and build up with some examples in python along the way. We will do the whole algorithm from scratch!

## The Base - What do we use Logistic Regression for?

If you are first understanding logistic regression, you have to know one thing. You have to predict one of two classes, like cats vs dogs, fire vs. ice, etc. 
In data science, we typically give what we are trying to predict a name. We will call them labels, or <!-- $Y$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=Y">. We have to codify the labels to be some number. 
Again, we typically give our <!-- $Y$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=Y"> the values 1 or 0, 1 corresponding to one label, and 0 for the other label.

<div style="text-align:center"><img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+Y+%3D%0A%5Cbegin%7Bcases%7D%0A1+%5C%5C%0A0%0A%5Cend%7Bcases%7D" 
alt="Y =
\begin{cases}
1 \\
0
\end{cases}"></div>

So we have a bunch of labels <!-- $Y$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=Y">. How do we predict one of those classes? Well, you gotta have some data that represents each class. We typically call these features, or <!-- $X$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=X">.

Features represent the labels. For classifying cats and dogs, we may have one feature called color with many different color values (red, yellow, orange). 
We might have another feature called length, with varying lengths of the animal (20 inches, 50 inches, etc.). 
As you gather more features describing cats and dogs, you are developing a matrix that will have a bunch of columns (features) and rows (the values for one animal). 

Let's move to another more complicated example. I have a dataset downloaded from kaggle called the South Africa Coronary Heart Disease Dataset {https://www.kaggle.com/emilianito/saheart},or SAHeart. 
We are going to be predicting whether or not a sample of males from South Africa have coronary heart disease based on a host of measured features, including systolic blood pressure, tobacco use, LDL cholesterol, adiposity, family history, type-A behavior, obesity, alcohol use, and age.

Each individual feature can be denoted as <!-- $x_i$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x_i">. For example, <!-- $x_1$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x_1"> represent the data in the "adiposity" feature in the SA Heart Dataset. 

We have about 462 rows, which means we have measured 462 patients. With this data, you could "train" an algorithm to predict if the person has coronary heart disease, 1 for "yes they have it", and 0 for "no they do not have it."

## How do we predict a class?

So you're probably wondering how do we get from a set of features, <!-- $X$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=X">, to a set of predictions, <!-- $Y$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=Y">? We have to do a little bit of math. We need some equation to be able to take the features <!-- $X$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=X"> and get an output of 1 or 0. This equation will basically be "fitting" the numerical features to the numerical outcome (1 or 0). 
That means when a new set of features comes around, we can just use the fit equation to predict 1 or 0. Let's take a look at how this all works.

We need to define some notation first. Let's define the probability that a given set of features is 1.

$$ P(Y=1) = p $$

Now let's define the probability that a given set of features is 0.

$$ P(Y=0)= 1 - p $$

When we are doing logistic regression, we are predicting a probability, between 1 and 0. We will be using p for when the predicted class is 1, and 1 - p when the predicted class is 0.

So now let's take a look at something interesting. We can use p and 1-p to get the odds that a class is 1 or 0 with this equation here.

$$\frac{p} {1-p} = \frac{P(Y=1)} {P(Y=0)}$$

This is known as the odds, which is p divided by 1-p. When the odds are 1, you have equal probability of getting a class. When odds are greater than 1, you will predict the class is 1. If less than 1, you will predict the class is 0. This is all well and good for understanding the relationships between probabilities, but it still doesn't help us get from features to predictions. We need something more.

Here comes the binary [logit function](https://en.wikipedia.org/wiki/Logit). This was formulated some time ago and has been critical in linking the probability values 1 and 0 to our features X. Odds are calculated above and below 1. However, if you take the log of the odds, you get the log odds, which maps values between 0 and 1. Logistic regression suggests a linear relationship between the features, X, and the log odds of the the event that Y=1,which is p. 

You can map the total relationship between the log odds and your features below.

$$logit(p) = \log{\left(\frac{p} {1-p}\right)} = log(p) - log(1-p) = -log\left(\frac{1} {p} - 1\right)$$

$$-log\left(\frac{1} {p} - 1\right) = \beta_{0} + \beta_{1}x_{1} + \dots + \beta_{p-1}x_{p-1} = \beta^TX$$


, where 
- $p$ where $y=1$ is so awesome
- <!-- $p$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p"> is the probability that <!-- $y=1$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=y%3D1">,
- <!-- $\beta_{1}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cbeta_%7B1%7D"> to <!-- $\beta_{p-1}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cbeta_%7Bp-1%7D"> are the weights/coefficients mapped to your features (each beta is one value) 
- <!-- $x_{1}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x_%7B1%7D"> to <!-- $x_{p-1}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x_%7Bp-1%7D"> are each individual feature vector (vector (many values) for adiposity, etc.)
- <!-- $\beta_{0}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cbeta_%7B0%7D"> is an intercept.

<!-- $\beta^TX$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cbeta%5ETX"> are the weights and the features condensed into matrix form. You can do some cool stuff with this property. I will show that later in later equations.

So maybe you are starting to understand how we map outputs to features. It's all in the Betas! Those are the weights or coefficients "attached" to your numerical features. The logistic equation is trying to adjust those weights in the best possible based on all the data for a particular feature to give the best output for <!-- $Y$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=Y">.

Now that we know this relationship between the features and the probabilities, how do we actually get the probability that y=1? We have to take that logit equation and exponentiate it. This will give us everything we need to make a prediction!

Step 1: Exponentiate the log odds to get the odds, but now we have the features exponentiated as well.

$$\frac{p} {1-p} = e^{ \beta_{0} + \beta_{1}x_{1} + \dots + \beta_{p-1}x_{p-1} }$$

Step 2: Solve for p

$$p = \frac{ e^{ \beta_{0} + \beta_{1}x_{1} + \dots + \beta_{p-1}x_{p-1}  } } {1 + e^{ \beta_{0} + \beta_{1}x_{1} + \dots + \beta_{p-1}x_{p-1}  }} = \frac{ e^{\beta^TX} } {1 + e^{\beta^TX}}$$

Step 3: Simplify

$$p = \frac{ 1 } {1 + e^{-\beta_{0} + \beta_{1}x_{1} + \dots + \beta_{p-1}x_{p-1}  }} = \frac{ 1 } {1 + e^{-\beta^TX }}$$


This is it! If you know the Betas and you have your feature values, all you have to do is plug and chug to get a prediction for the probability that y=1 (p). 
Let's see what this looks like in python code. This final equation is called the sigmoid function, where the matrix multiplication of your Betas and your features X is typically "z" in a lot of things you will read online.

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
We will have to do some optimization.

## How do we get those coefficients to make an accurate prediction?

How do we get the best weights that fit the data in the most accurate way? We have to do some optimization or maximum likelihood estimation!

What the heck is that? In the most simple terms, you need to find the results on your data that best minimize the error in your predictions from the actual class.
This must be done by just picking weights and seeing what happens. Then you update them and repeat. When you get the best weights you can, you stick with them. 
This is what all the model training is. Let's get into it.


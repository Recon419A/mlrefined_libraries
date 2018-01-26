import autograd.numpy as np
from autograd import value_and_grad 
from autograd import hessian
from autograd.misc.flatten import flatten_func
import copy
from inspect import signature


'''
A list of cost functions for supervised learning.  Use the choose_cost function
to choose the desired cost with input data  
'''
class Setup:
    def __init__(self,x,y,feature_transforms,cost,**kwargs):
        normalize = 'standard'
        if 'normalize' in kwargs:
            normalize = kwargs['normalize']
        if normalize == 'standard':
            # create normalizer
            self.normalizer = self.standard_normalizer(x)

            # normalize input 
            self.x = self.normalizer(x)
        elif normalize == 'sphere':
            # create normalizer
            self.normalizer = self.PCA_sphereing(x)

            # normalize input 
            self.x = self.normalizer(x)
        else:
            self.x = x
            self.normalizer = lambda data: data
            
        # make any other variables not explicitly input into cost functions globally known
        self.y = y
        self.feature_transforms = feature_transforms
        
        # count parameter layers of input to feature transform
        self.sig = signature(self.feature_transforms)

        self.lam = 0
        if 'lam' in kwargs:
            self.lam = kwargs['lam']

        # make cost function choice
        cost_func = 0
        if cost == 'least_squares':
            self.cost_func = self.least_squares
        if cost == 'least_absolute_deviations':
            self.cost_func = self.least_absolute_deviations
        if cost == 'softmax':
            self.cost_func = self.softmax
        if cost == 'relu':
            self.cost_func = self.relu
        if cost == 'counter':
            self.cost_func = self.counting_cost
        if cost == 'multiclass_perceptron':
            self.cost_func = self.multiclass_perceptron
        if cost == 'multiclass_softmax':
            self.cost_func = self.multiclass_softmax
        if cost == 'multiclass_counter':
            self.cost_func = self.multiclass_counting_cost

    # run optimization
    def fit(self,**kwargs):
        # basic parameters for gradient descent run
        max_its = 500; alpha_choice = 10**(-1);
        w = 0.1*np.random.randn(np.shape(self.x)[0] + 1,1)

        # set parameters by hand
        if 'max_its' in kwargs:
            max_its = kwargs['max_its']
        if 'alpha_choice' in kwargs:
            alpha_choice = kwargs['alpha_choice']
        if 'w' in kwargs:
            w = kwargs['w']

        # run gradient descent
        self.weight_history, self.cost_history = self.gradient_descent(self.cost_func,alpha_choice,max_its,w)

    ###### cost functions #####
    # compute linear combination of input point
    def model(self,x,w):   
        # feature transformation - switch for dealing
        # with feature transforms that either do or do
        # not have internal parameters
        f = 0
        if len(self.sig.parameters) == 2:
            if np.shape(w)[1] == 1:
                f = self.feature_transforms(x,w)
            else:
                f = self.feature_transforms(x,w[0])
        else: 
            f = self.feature_transforms(x)    

        # tack a 1 onto the top of each input point all at once
        o = np.ones((1,np.shape(f)[1]))
        f = np.vstack((o,f))

        # compute linear combination and return
        # switch for dealing with feature transforms that either 
        # do or do not have internal parameters
        a = 0
        if np.ndim(w) == 2:
            a = np.dot(f.T,w)
        elif np.ndim(w) == 3:
            a = np.dot(f.T,w[1])
        return a
    
    # an implementation of the least squares cost function for linear regression
    def least_squares(self,w):
        cost = np.sum((self.model(self.x,w) - self.y)**2)
        return cost/float(len(self.y))

    # a compact least absolute deviations cost function
    def least_absolute_deviations(self,w):
        cost = np.sum(np.abs(self.model(self.x,w) - self.y))
        return cost/float(len(self.y))

    # the convex softmax cost function
    def softmax(self,w):
        cost = np.sum(np.log(1 + np.exp(-self.y*self.model(self.x,w))))
        return cost/float(len(self.y))

    # the convex relu cost function
    def relu(self,w):
        cost = np.sum(np.maximum(0,-self.y*self.model(self.x,w)))
        return cost/float(len(self.y))

    # the counting cost function
    def counting_cost(self,w):
        cost = np.sum((np.sign(self.model(self.x,w)) - self.y)**2)
        return 0.25*cost 

    # multiclass perceptron
    def multiclass_perceptron(self,w):        
        # pre-compute predictions on all points
        all_evals = self.model(self.x,w)

        # compute maximum across data points
        a = np.max(all_evals,axis = 1)        

        # compute cost in compact form using numpy broadcasting
        b = all_evals[np.arange(len(self.y)),self.y.astype(int).flatten()]
        cost = np.sum(a - b)

        # add regularizer
        cost = cost + self.lam*np.linalg.norm(w[1:,:],'fro')**2

        # return average
        return cost/float(len(self.y))

    # multiclass softmax
    def multiclass_softmax(self,w):        
        # pre-compute predictions on all points
        all_evals = self.model(self.x,w)

        # compute softmax across data points
        a = np.log(np.sum(np.exp(all_evals),axis = 1)) 

        # compute cost in compact form using numpy broadcasting
        b = all_evals[np.arange(len(self.y)),self.y.astype(int).flatten()]
        cost = np.sum(a - b)

        # add regularizer
        cost = cost + self.lam*np.linalg.norm(w[1:,:],'fro')**2

        # return average
        return cost/float(len(self.y))

    # multiclass misclassification cost function - aka the fusion rule
    def multiclass_counting_cost(self,w):                
        # pre-compute predictions on all points
        all_evals = self.model(self.x,w)

        # compute predictions of each input point
        y_predict = (np.argmax(all_evals,axis = 1))[:,np.newaxis]

        # compare predicted label to actual label
        count = np.sum(np.abs(np.sign(self.y - y_predict)))

        # return number of misclassifications
        return count
    
    ##### optimizer ####
    # gradient descent function - inputs: g (input function), alpha (steplength parameter), max_its (maximum number of iterations), w (initialization)
    def gradient_descent(self,g,alpha_choice,max_its,w):
        # compute the gradient function of our input function - note this is a function too
        # that - when evaluated - returns both the gradient and function evaluations (remember
        # as discussed in Chapter 3 we always ge the function evaluation 'for free' when we use
        # an Automatic Differntiator to evaluate the gradient)
        gradient = value_and_grad(g)

        # run the gradient descent loop
        weight_history = []      # container for weight history
        cost_history = []        # container for corresponding cost function history
        alpha = 0
        for k in range(1,max_its+1):
            # check if diminishing steplength rule used
            if alpha_choice == 'diminishing':
                alpha = 1/float(k)
            else:
                alpha = alpha_choice

            # evaluate the gradient, store current weights and cost function value
            cost_eval,grad_eval = gradient(w)
            weight_history.append(w)
            cost_history.append(cost_eval)

            # take gradient descent step
            w = w - alpha*grad_eval

        # collect final weights
        weight_history.append(w)
        # compute final cost function value via g itself (since we aren't computing 
        # the gradient at the final step we don't get the final cost function value 
        # via the Automatic Differentiatoor) 
        cost_history.append(g(w))  
        return weight_history,cost_history
    
    ###### normalizers #####
    # standard normalization function 
    def standard_normalizer(self,x):
        # compute the mean and standard deviation of the input
        x_means = np.mean(x,axis = 1)[:,np.newaxis]
        x_stds = np.std(x,axis = 1)[:,np.newaxis]   

        # create standard normalizer function
        normalizer = lambda data: (data - x_means)/x_stds

        # return normalizer 
        return normalizer

    # compute eigendecomposition of data covariance matrix
    def PCA(self,x,**kwargs):
        # regularization parameter for numerical stability
        lam = 10**(-7)
        if 'lam' in kwargs:
            lam = kwargs['lam']

        # create the correlation matrix
        P = float(x.shape[1])
        Cov = 1/P*np.dot(x,x.T) + lam*np.eye(x.shape[0])

        # use numpy function to compute eigenvalues / vectors of correlation matrix
        D,V = np.linalg.eigh(Cov)
        return D,V

    # PCA-sphereing - use PCA to normalize input features
    def PCA_sphereing(self,x,**kwargs):
        # standard normalize the input data
        standard_normalizer = self.standard(x)
        x_standard = standard_normalizer(x)
        
        # compute pca transform 
        D,V = self.PCA(x_standard,**kwargs)
        
        # compute forward sphereing transform
        D_ = np.array([1/d**(0.5) for d in D])
        D_ = np.diag(D_)
        W = np.dot(D_,V.T)
        pca_sphere_normalizer = lambda data: np.dot(W,standard_normalizer(data))

        # return normalizer 
        return pca_sphere_normalizer
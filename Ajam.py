#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from numpy import pi


# # Q1: Coin toss
# 
# Read section 2.1 of Sivia and recreate figure 2.3.

# ## (a)
# 
# Define the prior functions

# In[41]:


def uniform_prior(H):
    if  H > 1 or H < 0:
        return 0
    
    return 1
    
first_prior = np.vectorize(uniform_prior)
    

    

def gaussian_prior(H):
    if  H > 1 or H < 0:
        return 0
    else:
        sigma=0.03
        mu=0.5
        F=(1/(2*np.pi*sigma**2))*(np.exp(-1*(H-mu)**2)/2*sigma**2)
        return F
second_prior = np.vectorize(gaussian_prior)


def power20_prior(H):
    if H > 1 or H < 0:
        return 0
    return (H - 0.5) ** 20

third_prior = np.vectorize(power20_prior)




# ### Bonus point
# 
# Why do we need to use np.vectorize here?

# ## (b)
# 
# Create an array for H, calculte the priors, normalize them and then plot them.

# In[48]:


H = np.linspace(0, 1, 1000, dtype=np.float64)


# In[46]:


plt.figure(figsize=(15, 3))
plt.suptitle('Three different priors (Normalized)')

firstprior = first_prior(H) / np.max(first_prior(H))
plt.subplot(1, 3, 1)
plt.plot(H, firstprior)
#Use np.max in order to normalization

secondprior = second_prior(H) / np.max(second_prior(H))
plt.subplot(1, 3, 2)
plt.plot(H, secondprior)

thirdprior = third_prior(H) / np.max(third_prior(H))
plt.subplot(1, 3, 3)
plt.plot(H, thirdprior)

plt.show()


# ## (c)
# 
# Here we explicitly declared a data type for the array H. Although most of the times python does it for us, here was an example that we need to do it by hand to avoid round-off error. Find out what this error is and write a few lines about it (In Farsi or English) and explain what did we do to avoid it here. In other words, why did we use 'np.float128' exept just using 'float'?
# The error is module 'numpy' has no attribute 'float128'
# And the Suggestion is to use np.longdouble in place of np.float128
# 
# You can also emit the data type decleration from the definition of H and see what happens in calculating the posterior.

# 

# ## (d)
# 
# Write a proper function to calculate the psterior for a given data and recreate the given plot.

# In[52]:


data = [np.random.choice([0, 0, 0, 0, 0, 0, 0, 1, 1, 1], size=(i)) for i in range(4)]  
data.extend([np.random.choice([0, 0, 0, 0, 0, 0, 0, 1, 1, 1], size=(2**i)) for i in range(2, 13)])


# In[50]:


def posterior(H, data, prior):
    ones = np.sum(data)
    total = len(data)
    return H ** ones * (1 - H) ** (total - ones) * prior


# In[57]:


plt.figure(figsize=(15, 15))


for i in range(15):
    plt.subplot(5, 3, i + 1)
    post1 = posterior(H, data[i], firstprior)
    post1 = post1 / np.max(firstprior)
    plt.plot(H, post1)
    
    post2 = posterior(H, data[i], secondprior)
    post2 = post2 / np.max(secondprior)
    plt.plot(H, post2)
    
    post3 = posterior(H, data[i], thirdprior)
    post3 = post3 / np.max(thirdprior)
    plt.plot(H, post3)
    
    plt.text(0.95, 0.85, str(len(data[i])),
             fontsize=15, ha = 'center')
    
plt.tight_layout(h_pad=3)
plt.show()


# # Q2: Distributions and moments

# ## (a)
# 
# Write a function to calculate raw moments of a given distribution 

# In[58]:


def moment(data, n):
    mu = 0
    number = 0
    for i in data:
        mu += i ** n
        num += 1
    return mu / number


# ## (b)
# 
# Write a function to calculate central moments of a given distribution

# In[59]:


def central_moment(data, n):
    mu_1 = moment(data, 1)
    cen_nu= 0
    num = 0
    for i in data:
        cen_mu += (i - mu_1) ** n
        num += 1
    return cen_mu / num


# ## (c)
# 
# Using numpy.random, create a binomial (p=0.7, n=$30$), a Poisson ($\mu$ = 2.1) and a gaussian ($\mu$ = 2.1, $\sigma = 0.3$) distribution with size $10^6$ and calculate their first 3 raw and central moments

# In[62]:


bio = np.random.binomial(p = 0.7, n = 30, size = 10 ** 6)
poi = np.random.poisson(lam = 2.1, size = 10 ** 6)
gus = np.random.normal(loc = 2.1,scale = 0.3, size = 10 ** 6)


# ## (d)
# 
# Using scipy.stats, calculate the first three moments of these distributions and check with the results from your own function. What does the function in scipy calculate? Raw moments or central moments?

# In[60]:


from scipy import stats


# In[66]:


print('bio(central):', '\n\t1st: ', stats.moment(bino, moment = 1), '\n\t2nd: ', stats.moment(bino, moment = 2), '\n\t3rd: ', stats.moment(bino, moment = 3))
print('poi(central):', '\n\t1st: ', stats.moment(poiss, moment = 1), '\n\t2nd: ', stats.moment(poiss, moment = 2), '\n\t3rd: ', stats.moment(poiss, moment = 3))
print('gus(central):', '\n\t1st: ', stats.moment(gauss, moment = 1), '\n\t2nd: ', stats.moment(gauss, moment = 2), '\n\t3rd: ', stats.moment(gauss, moment = 3))


# # Q3: Radioactive decay

# The Poisson distribution is often used to describe situations in which an event occurs repeatedly at a constant rate of probability. An application of this distribution involves the decay of radioactive samples, but only in the approximation that the decay rate is slow enough that depletion in the population of the decaying species can be neglected.
# 
# Now suppose we have a data set showing the number of $\alpha$ particles emmited in 7.5 sec intervals:

# ## (a)
# 
# Load the provided data into a pandas data frame and show the data and then plot it's PDF.

# In[72]:


from pandas import DataFrame, read_csv


# In[77]:


Data = pandas.read_csv('data.txt', delimiter = '\t')


# ## (b)
# 
# Calculate total number of decays and the average number of decays after each time interval. Then add the data to the Pandas DataFrame and show it. Then save the data to file 'new_data.txt'.

# In[ ]:





# ## (c)
# 
# Use a unifor prior and do the same analysis as question 1 to get the mean decay number per interval. Plot the diagrams as well. (Beware that this time we have a Poisson distribution rather than a binomial one)

# In[ ]:





# ## (d)
# 
# Use two different priors of your choise and repeat the analysis. Which of these three priors leads to an answer sooner?

# In[ ]:





# ## (e)
# 
# Now that you have the average decay per time interval, calculate the half life of this radioactive element (Suppose that the sample is large enough not to lose a noticable fraction of the particles in the span of the experiment). Can find out what this element is?

# In[ ]:





# # Bonus question
# 
# Do an error analysis and report how sure are you about the number you derived for the half life of the element.

# In[ ]:





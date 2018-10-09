
# coding: utf-8

# # Multi-layer Perceptron
# 
# ## Lab-Assignment 6
# 
# - P.Aneesh
# - 16BCE1037
# 
# ### Abstract:
# A multilayer perceptron (MLP) is a class of feedforward artificial neural network. An MLP consists of at least three layers of nodes. Except for the input nodes, each node is a neuron that uses a nonlinear activation function. MLP utilizes a supervised learning technique called backpropagation for training. Its multiple layers and non-linear activation distinguish MLP from a linear perceptron. It can distinguish data that is not linearly separable.
# Multilayer perceptrons are sometimes colloquially referred to as "vanilla" neural networks, especially when they have a single hidden layer.
# 
# 
# ### Methodology:
# Multi-Layer Perceptrons:
# The field of artificial neural networks is often just called neural networks or multi-layer perceptrons after perhaps the most useful type of neural network. A perceptron is a single neuron model that was a precursor to larger neural networks.
# It is a field that investigates how simple models of biological brains can be used to solve difficult computational tasks like the predictive modeling tasks we see in machine learning. The goal is not to create realistic models of the brain, but instead to develop robust algorithms and data structures that we can use to model difficult problems.
# The power of neural networks come from their ability to learn the representation in your training data and how to best relate it to the output variable that you want to predict. In this sense neural networks learn a mapping. Mathematically, they are capable of learning any mapping function and have been proven to be a universal approximation algorithm.
# The predictive capability of neural networks comes from the hierarchical or multi-layered structure of the networks. The data structure can pick out (learn to represent) features at different scales or resolutions and combine them into higher-order features. For example from lines, to collections of lines to shapes.

# ## Dataset: 
# 
# Given dataset is the wheatSeed dataset which has the following column 
# 
# - area 
# - perimeter
# - compactness
# - lengthOfKernel
# - widthOfKernel
# - asymmetryCoefficient
# - lengthOfKernelGroove
# - TypeOfWheatSeed

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score


# In[2]:


col_Names=["area", "perimeter", "compactness", "lengthOfKernel", "widthOfKernel", "asymmetryCoefficient", "lengthOfKernelGroove", "TypeOfWheatSeed"]
data = pd.read_csv("seeds_dataset.csv",names=col_Names, header = None)


# In[3]:


data.head()


# In[4]:


data.isnull().any()


# In[5]:


X = data[["area","perimeter","compactness","lengthOfKernel","widthOfKernel","asymmetryCoefficient","lengthOfKernelGroove"]].values


# In[6]:


Y = data["TypeOfWheatSeed"].values


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)  


# In[8]:


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[9]:


mlpSeed = MLPClassifier(hidden_layer_sizes = (13,13,13), max_iter = 500)


# In[10]:


mlpSeed.fit(X_train, y_train)   


# In[11]:


#printing each layers final weights
for i,j in enumerate(mlpSeed.coefs_):
    print("Final Weights for %d layer:\n" %(i+1))
    for ind,val in enumerate(j):
        print("Weights for %d neuron:" %(ind+1))
        print(val)


# In[12]:


predictions = mlpSeed.predict(X_test)


# In[13]:


print("Accuracy of predictions: %.2f" %(accuracy_score(y_test, predictions) * 100))
print("Classification Report for the MLP: ")
print(classification_report(y_test, predictions))


# ## Accuracy/Epoch

# In[14]:


graphy=[]
graphx=[]
for epoch in range(250,10000,250):
    mlp = MLPClassifier(hidden_layer_sizes = (15,20,15), max_iter = epoch)
    mlp.fit(X_train, y_train)
    predictions = mlp.predict(X_test)
    graphy.append(accuracy_score(predictions, y_test))
    graphx.append(epoch)


# In[15]:


d= {'Epoch': graphx,'Accuracy':graphy}


# In[16]:


plt.plot("Epoch","Accuracy",data=d)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')


# In[17]:


print("Number of epoches  accuracy")
for i in range(0,len(graphx)):
    print(graphx[i],"\t",graphy[i])


# ### We an now clearly see that the accuracy is changing based on number of epochs
# ### We now try changing the number of nodes

# In[18]:


graphy=[]
graphx=[]
for nodes in range(1,20,1):
    mlp = MLPClassifier(hidden_layer_sizes = (nodes,nodes,nodes), max_iter = 5000)
    mlp.fit(X_train, y_train)
    predictions = mlp.predict(X_test)
    graphy.append(accuracy_score(predictions, y_test))
    graphx.append(nodes)


# In[19]:


plt.plot(graphx,graphy)
plt.ylabel('Accuracy')
plt.xlabel('Node')


# In[20]:


print("Number of nodes  accuracy")
for i in range(0,len(graphx)):
    print(graphx[i],"\t",graphy[i])


# ### Now we'll try to add more layers to this and see how our accuracy stands up
# 
# Now each layers will have 5 nurons/nodes

# In[21]:


graphy=[]
graphx=[]
layers=[5]
for nodes in range(1,10,1):
    mlp = MLPClassifier(hidden_layer_sizes = (layers), max_iter = 5000)
    mlp.fit(X_train, y_train)
    predictions = mlp.predict(X_test)
    graphy.append(accuracy_score(predictions, y_test))
    graphx.append(len(layers))
    layers.append(5)


# In[22]:


plt.plot(graphx,graphy)
plt.ylabel('Accuracy')
plt.xlabel('Layers')


# In[23]:


print("Layers  accuracy")
for i in range(0,len(graphx)):
    print(graphx[i],"\t",graphy[i])


# ## Result:
# 
# Hence we have seen the effect of accuracy on changing the number of layers, number of nodes in a layer and output nodes and also plotted the graphes.¶

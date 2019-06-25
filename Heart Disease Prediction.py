
# # Heart Disease Prediction
# 
# In this machine learning project, We have collected the dataset from Kaggle and We will be using Machine Learning to make predictions on whether a person is suffering from Heart Disease or not.

# ### Import libraries
# 
# Let's first import all the necessary libraries. We'll use `numpy` and `pandas` to start with. For visualization, We will use `pyplot` subpackage of `matplotlib`, use `rcParams` to add styling to the plots and `rainbow` for colors. For implementing Machine Learning models and processing of data, We will use the `sklearn` library.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
get_ipython().magic(u'matplotlib inline')
import warnings
warnings.filterwarnings('ignore')


# For processing the data, We'll import a few libraries. To split the available dataset for testing and training, We'll use the `train_test_split` method. To scale the features, We are using `StandardScaler`.

# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Next, We'll import all the Machine Learning algorithms we will be using.
# 1. K Neighbors Classifier
# 
# 2. Decision Tree Classifier
# 

# In[3]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# ### Importing dataset
# 
# Now that we have all the libraries we will need, we can import the dataset and take a look at it. The dataset is stored in the file `dataset.csv`. We'll use the pandas `read_csv` method to read the dataset.

# In[4]:


dataset = pd.read_csv('dataset.csv')


# The dataset is now loaded into the variable `dataset`. We'll just take a glimpse of the data using the `desribe()` and `info()` methods before We actually start processing and visualizing it.

# In[5]:


dataset.info()


# Looks like the dataset has a total of 303 rows and there are no missing values. There are a total of `13 features` along with one target value which we wish to find.

# In[6]:


dataset.describe()


# The scale of each feature column is different and quite varied as well. While the maximum for `age` reaches 77, the maximum of `chol` (serum or high cholesterol) is 564.

# ### Understanding the data
# 
# Now, we can use visualizations to better understand our data and then look at any processing we might want to do.

# In[7]:


rcParams['figure.figsize'] = 20, 14
plt.matshow(dataset.corr())
plt.yticks(np.arange(dataset.shape[1]), dataset.columns)
plt.xticks(np.arange(dataset.shape[1]), dataset.columns)
plt.colorbar()


# Taking a look at the correlation matrix above, it's easy to see that a few features have negative correlation with the target value while some have positive.
# Next, We'll take a look at the histograms for each variable.

# In[8]:


dataset.hist()


# Taking a look at the histograms above, We can see that each feature has a different range of distribution. Thus, using scaling before our predictions should be of great use. Also, the categorical features do stand out.

# It's always a good practice to work with a dataset where the target classes are of approximately equal size. Thus, let's check for the same.

# In[9]:


rcParams['figure.figsize'] = 8,6
plt.bar(dataset['target'].unique(), dataset['target'].value_counts(), color = ['red', 'green'])
plt.xticks([0, 1])
plt.xlabel('Target Classes')
plt.ylabel('Count')
plt.title('Count of each Target Class')


# The two classes are not exactly 50% each but the ratio is good enough to continue without dropping/increasing our data.

# ### Data Processing
# 
# After exploring the dataset, We observed that we need to convert some categorical variables into dummy variables and scale all the values before training the Machine Learning models.
# First, We'll use the `get_dummies` method to create dummy columns for categorical variables.

# In[10]:


dataset = pd.get_dummies(dataset, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])


# Now, We will use the `StandardScaler` from `sklearn` to scale the dataset.

# In[11]:


standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])


# The data is not ready for our Machine Learning application.

# ### Machine Learning
# 
# We'll now import `train_test_split` to split our dataset into training and testing datasets. Then, We'll import all Machine Learning models we'll be using to train and test the data.

# In[12]:


y = dataset['target']
X = dataset.drop(['target'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)


# #### K Neighbors Classifier
# 
# The classification score varies based on different values of neighbors that we choose. Thus, we'll plot a score graph for different values of K (neighbors) and check when do we achieve the best score.

# In[13]:


knn_scores = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(X_train, y_train)
    knn_scores.append(knn_classifier.score(X_test, y_test))


# We have the scores for different neighbor values in the array `knn_scores`. We'll now plot it and see for which value of K did we get the best scores.

# In[14]:


plt.plot([k for k in range(1, 21)], knn_scores, color = 'red')
for i in range(1,21):
    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
plt.xticks([i for i in range(1, 21)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')


# From the plot above, it is clear that the maximum score achieved was `0.87` for the 8 neighbors.

# In[15]:


print("The score for K Neighbors Classifier is {}% with {} nieghbors.".format(knn_scores[7]*100, 8))


# #### Decision Tree Classifier
# 
# Here, we'll use the Decision Tree Classifier to model the problem at hand. We'll vary between a set of `max_features` and see which returns the best accuracy.

# In[16]:


dt_scores = []
for i in range(1, len(X.columns) + 1):
    dt_classifier = DecisionTreeClassifier(max_features = i, random_state = 0)
    dt_classifier.fit(X_train, y_train)
    dt_scores.append(dt_classifier.score(X_test, y_test))


# We selected the maximum number of features from 1 to 30 for split. Now, let's see the scores for each of those cases.

# In[17]:


plt.plot([i for i in range(1, len(X.columns) + 1)], dt_scores, color = 'green')
for i in range(1, len(X.columns) + 1):
    plt.text(i, dt_scores[i-1], (i, dt_scores[i-1]))
plt.xticks([i for i in range(1, len(X.columns) + 1)])
plt.xlabel('Max features')
plt.ylabel('Scores')
plt.title('Decision Tree Classifier scores for different number of maximum features')


# The model achieved the best accuracy at three values of maximum features, `2`, `4` and `18`.

# In[18]:


print("The score for Decision Tree Classifier is {}% with {} maximum features.".format(dt_scores[17]*100, [2,4,18]))


# ### Conclusion
# 
# In this project, We used Machine Learning to predict whether a person is suffering from a heart disease. After importing the data, we analysed it using plots. Then, we generated dummy variables for categorical features and scaled other features. 
# We, then, applied four Machine Learning algorithms, `K Neighbors Classifier` and `Decision Tree Classifier`. We varied parameters across each model to improve their scores.
# In the end, `K Neighbors Classifier` achieved the highest score of `87%` with `8 nearest neighbors`.
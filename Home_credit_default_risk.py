#!/usr/bin/env python
# coding: utf-8

# # HOME_CREDIT_DEFAULT_RISK

# Many people struggle to get loans due to insufficient or non-existent credit histories. And, unfortunately, this population is often taken advantage of by untrustworthy lenders.
# 
# 
# Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. In order to make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data--including telco and transactional information--to predict their clients' repayment abilities.
# 
# While Home Credit is currently using various statistical and machine learning methods to make these predictions, they're challenging Kagglers to help them unlock the full potential of their data. Doing so will ensure that clients capable of repayment are not rejected and that loans are given with a principal, maturity, and repayment calendar that will empower their clients to be successful.
# 
# 

# Home Credit official site is <a href="https://www.homecredit.co.in/en?msclkid=915d570ab3e611ec809c3ac29fce3a0d"> here </a>. Check this website to understand more on Home_credit

# _____

# So before we start have a basic understanding on <a href="https://medium.com/analytics-vidhya/supervised-learning-d8562826b798">Supervised</a> and <a href="https://medium.com/analytics-vidhya/beginners-guide-to-unsupervised-learning-76a575c4e942">unsupervised</a> learning.
# 
# In this project mainly we have to create a model which predicts, that a person can be availed a loan or not. So this comes under classification and classification comes under supervised learning. 

# _____

# In[2]:


#Now let us import the dataset to our jupyter notebook
import pandas as pd #Pandas is used for data exploration (EDA)
import numpy as np # NumPy is a short form for Numerical Python, which is applied for scientific programming in Python, especially for numbers
import matplotlib.pyplot as plt # Matplotlib is a plotting library 
import seaborn as sns #Seaborn is a popular data visualization library for Python
from sklearn.preprocessing import LabelEncoder #Is a way to encode class levels.
import warnings #This is to remove the warning messages that we get so that the notebook looks clean
warnings.filterwarnings('ignore') #filter in Python handles warnings


# ###### Here are some tutorials. Kindly go the below blogs and understand more about the topics
# ____
# <a href="https://www.tutorialspoint.com/python_pandas/python_pandas_introduction.htm">Pandas</a>
# 
# <a href="https://pandas.pydata.org/docs/">Pandas Documentation</a>
# 
# <a href="https://numpy.org/doc/stable/user/whatisnumpy.html">Numpy</a>
# 
# <a href="https://www.geeksforgeeks.org/python-introduction-matplotlib/">matplotlib</a>
# 
# <a href="https://seaborn.pydata.org/introduction.html">Seaborn</a>
# 
# <a href="https://vitalflux.com/labelencoder-example-single-multiple-columns/">What is Encoding? </a>

# _______

# In[3]:


# Now we have impported nessasary modules and now let us import the data to the jupyter notebook
data = pd.read_csv('application_train.csv') # read_csv is a pandas function which will help you to import the date here


# In[4]:


data


# _________________________

# From the above we see the over view of the dataset.

# _____________________

# In[5]:


data.head() #head fuction will help you see the first 5 rows to understand the headers and to have an overview. 


# In[6]:


data.shape #The shape command will show the rows and columns of the dataset


# In[7]:


data.info()# info command will give all the information about this dataset


# In[8]:


data.describe() #This function is used to generate descriptive statistics that summarize the central tendency.


# From the above table we can see the mean, count, std deviation etc

# Understand what is <a href="https://www.vedantu.com/commerce/mean-median-and-mode">Mean median mode</a>
# 
# Also understand <a href="https://www.investopedia.com/terms/s/standarddeviation.asp">Standard Deviation</a>

# Now let us see the target variable

# In[9]:


round(data['TARGET'].value_counts()/data.shape[0]*100, 2)


# From the above information we can clearly understand that, this is an imbalanced dataset. 
# 
# click <a href="https://medium.com/analytics-vidhya/what-is-balance-and-imbalance-dataset-89e8d7f46bc5"> here </a> under what is a balanced and imbalanced dataset.

# In[10]:


plt.hist(data['TARGET']) #let us also plot and see it


# In[11]:


data['TARGET'].unique() # unique formula will show the unique value of the dataset. 


# In[12]:


data.TARGET.unique() #We can also use dot and see the same result. The above syntax and this syntax gives same result. 


# Now let us identify the missing values in this dataset. Note: Most of the algorithm do not work with missing values. 

# In[13]:


data.isnull().sum() #here we are unable to see the dataset completely as there are lots of attribute here.


# Now let us change the option setting in jupyter notebook so that we can view the complete data here

# In[14]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# In[15]:


#Now let us rerun the data so that we can see the complete data here
data.isnull().sum()


# So now let us understand what is a <a href="https://medium.com/@vinitasilaparasetty/guide-to-handling-missing-values-in-data-science-37d62edbfdc1">Missing values</a>.
# 

# In[16]:


data.isnull().sum().sort_values(ascending=False)


# If the missing values are more than 75 percentage, then probably we can drop the feature as it does not contribute anything to the output

# In[17]:


(data.isnull().sum()/data.shape[0]*100).sort_values(ascending = False) #here we understood that non of them are >75%


# We are not going to drop any feature. Instead of that we will fill the null values later in this lecture. 

# Now let us make a table to understand more about the missing values

# In[18]:


def missing_val(dataset): 
    miss_val = dataset.isnull().sum()
    
    miss_val_percentage = 100 * dataset.isnull().sum() /len(dataset)
    
    miss_val_table = pd.concat([miss_val, miss_val_percentage], axis = 1)
    
    return miss_val_table


# Kindly <a href="https://www.bogotobogo.com/python/python_functions_def.php#:~:text=Here%20are%20brief%20descriptions%3A%201%20def%20is%20an,a%20result%20object%20back%20to%20the%20caller.%20?msclkid=b72b81e5b3fc11eca0abba3e964f117c">click</a> here to understand what is a def in Python. 

# In[19]:


missing_values = missing_val(data)


# In[20]:


missing_values = missing_values.rename(columns = {0: 'missing values', 1: '% of missing'}) #rename is a pandas fuction which is used to rename


# In[21]:


missing_values = missing_values.sort_values(by = '% of missing',ascending = False)


# In[22]:


missing_values


# As the indexs are replaced with the column name, we have changed it as shown below

# In[23]:


missing_values.reset_index(level=0, inplace=True)


# In[24]:


missing_values.rename(columns = {'index': 'column_name'}, inplace = True)


# In[25]:


missing_values


# We have created a decent table to understand the data structure. So in the above missing_val is a fuction which we created. 
# 
# If you are curious to know how to create a function in python click <a href="https://makitweb.com/how-to-create-functions-in-python/">here</a>

# So we have to impute the missing values because in most of the machine learning algorithm, we will have to fill these missing values. Like <a href="https://makitweb.com/how-to-create-functions-in-python/">XGBoost</a> can handle even if the dataset has missing values. 

# In[26]:


type(missing_values.describe()) 


# In[27]:


missing_values.describe() # so the attribute count is 122 here.


# Let us seperate the numerical and categorical data so that we can impute accordingly. 
# 
# The reason why we are seperating is because, we cannot fill with mean for the categorical data. 

# In[28]:


num = data.select_dtypes(exclude = "O") #here the select_dtypes is a pandas function which will help to select bool, string, int etc
obj = data.select_dtypes(include = "O")


# For the tutorial of select_dtypes please click <a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.select_dtypes.html?highlight=select_dtypes"> here </a> 

# In[29]:


num.head()


# In[30]:


num.isnull().sum().sort_values(ascending = False)


# Now let us fill the non value number with the mean for the numerical data. 

# In[31]:


num_mean = num.mean()
num_filled = num.fillna(num_mean) # This has helped to fill the missing values of multiple columns. It has filled the missing values with it's mean


# In[32]:


num_filled.isnull().sum().sort_values(ascending = True).head()


# In[33]:


obj.head() # as this is a categorical variable we shall fill this with the mode. 


# In[34]:


obj.isnull().sum().sort_values(ascending = False) #Let us fill with mode values.


# In[35]:


obj = obj.fillna(data.mode().iloc[0]) #Here we have filled the missing values for the categorical values


# In[36]:


obj.isnull().sum().head()


# # Now all the missing values are filled. Let us combine both the numerical and catergorical dataset

# In[37]:


df = pd.concat([obj, num_filled], axis = 1) #pandas concat will help to combine multiple data frames


# In[38]:


df.head()


# ## Now let us find the correction with the heatmap and table as well

# In[39]:


df.head(2)


# In[40]:


from matplotlib.pyplot import figure
figure(figsize=(20, 20), dpi=80)
sns.heatmap(df.corr())


# From the above heatmap we can understand that few are highly correlated which has to be removed to improve the model accuracy. 
# 
# FLAG_DOCUMENT_6
# 
# FLAG_DOCUMENT_8
# 
# FLAG_DOCUMENT_4
# 
# DAYS_BIRTH
# 
# The above columns will be removed.

# So if there is many correlation or multicollinearity we have to concider removing them. Kindly <a href="https://towardsdatascience.com/why-exclude-highly-correlated-features-when-building-regression-model-34d77a90ea8e">read</a>

# In[41]:


df = df.drop(['FLAG_DOCUMENT_6','FLAG_DOCUMENT_4','FLAG_DOCUMENT_8', 'DAYS_BIRTH'], axis = 1)


# In[42]:


len(df.columns)


# ### Also let us see those in a column

# In[43]:


correlation = df.corr() #If the dataset is huge we can check only by using this function beacause visualising has a limit on screen


# In[44]:


correlation


# ## Now let us find the Kurtosis and skewness, note: we are not removing those as for now

# In[45]:


df_num = df.select_dtypes(exclude = 'O')


# In[46]:


pos = []
neg = []

for i in df_num:
    if df_num[i].skew() >=1:
        print("The attribute", i.upper(), 'is positively skewed with a value of', df_num[i].skew())
        pos.append(i)
    elif df_num[i].skew() <= -1:
        print("The attribute", i.upper(), 'is negatively skewed with a value of', df_num[i].skew())
        neg.append(i)
    else:
        print('The attribute', i, 'is not a skewed and the value is: ', df_num[i].skew())


# <a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.append.html?ighlight=appen#pandas.DataFrame.append"> click here</a> to know about append function

# In[47]:


print(pos)


# In[48]:


print(neg)


# To understand what is skewness and Kurtosis <a href="https://makitweb.com/how-to-create-functions-in-python/">click here</a> here
# 

# In[50]:


kur = []
for i in df_num:
    print("The Kurtosis of column: ", i, 'is' ,df_num[i].kurt())
    kur.append(i)


# In[51]:


df.head(2)


# Let us remove the ouliers on hyper parameter tuning

# In[52]:


cat = df.select_dtypes(include = 'O')


# In[53]:


cat.columns


# In[54]:


for i in cat:
    print(i, len(cat[i].unique()))


# In[55]:


len(cat['ORGANIZATION_TYPE'].unique())


# In[56]:


pd.set_option('display.max_rows', 999)
pd.set_option('display.max_columns', 999)


# In[57]:


cat.info()


# In[58]:


cat = df.select_dtypes(include = "O")


# In[59]:


cat.value_counts()


# In[60]:


# Create a label encoder object
le = LabelEncoder()
le_count = 0

for col in df:
    if df[col].dtype == 'object':
        if len(list(df[col].unique())) <= 2:
            
            le.fit(df[col])
            df[col] = le.transform(df[col])
            le_count += 1


# In[61]:


cat = df.select_dtypes(include = "O")


# In[62]:


cat.head()


# In[63]:


cat = pd.get_dummies(cat)


# In[64]:


df = pd.concat([df_num, cat], axis = 1)


# In[65]:


df_num.shape[0] == cat.shape[0]


# After that let us scale and make the model. Later let us tune this by removing outlier if needed

# In[66]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
scaler_normal = MinMaxScaler()
scaler_rob = RobustScaler()


# Why do you wanted to scale a data? <a href="https://www.codementor.io/blog/scaling-ml-6ruo1wykxf#:~:text=%20Here%20are%20the%20inherent%20benefits%20of%20caring,as%20possible%20so%20that%20humans%20can...%20More%20?msclkid=b3e7e9e7b66911ec80238c95bf610935">click here</a> 
# 
# 

# # Now let us split the data to X and Y

# Why do we split the dataset? <a href="https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/#:~:text=The%20train-test%20split%20procedure%20is%20used%20to%20estimate,machine%20learning%20algorithms%20for%20your%20predictive%20modeling%20problem.?msclkid=dca127d5b66911ecbb64e9bcbfb37a5a">click here</a> 

# In[67]:


X = df.drop(['TARGET'], axis = 1)
y = df['TARGET']


# In[68]:


normal_scaled_X = scaler_normal.fit_transform(X)
robust_scaled_X = scaler_rob.fit_transform(X)


# In[72]:


y.head()


# In[69]:


from sklearn.preprocessing import StandardScaler


# In[70]:


scaler = StandardScaler()


# In[71]:


from imblearn.over_sampling import SMOTE 
from sklearn.metrics import confusion_matrix, classification_report 


# In[72]:


from sklearn.model_selection import train_test_split


# In[73]:


smt = SMOTE()


# To know more about SMOTE, kindly <a href="https://towardsdatascience.com/smote-fdce2f605729">click here</a>

# In[74]:


train_X, test_X, train_y, test_y = train_test_split(X,y, test_size= 0.25)


# In[75]:


X_train_res, y_train_res = smt.fit_sample(train_X, train_y) 


# In[76]:


from sklearn.linear_model import LogisticRegression


# In[77]:


lr1 = LogisticRegression() 
lr1.fit(X_train_res, y_train_res) 
predictions = lr1.predict(test_X) 
  
# print classification report 
print(classification_report(test_y, predictions))


# Do you wanted to know more on precision recall f1-score? <a href="https://medium.com/@mahesh.chavan1997/what-is-precision-recall-f1-score-b65b1965804c#:~:text=F1%20score%20gives%20the%20combined%20result%20of%20Precision,it%20is%20a%20complete%20failure%20of%20the%20model.?msclkid=23070936b66a11ecbdc5ef354bd74bfe">click here</a>

# ###### There are more technics called hyperparameter tuning and etc. Those will be taught in the next notebook

# In[ ]:





# In[ ]:





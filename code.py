#!/usr/bin/env python
# coding: utf-8

# ![](https://www.theglobeandmail.com/resizer/txoZ3yjElwytMy0k7ZhIeB_rGU8=/1500x1000/filters:quality(80):format(jpeg)/cloudfront-us-east-1.images.arcpublishing.com/tgam/YVOL2IWLZJOUPOZIX6OYZ3PHZY.jpg)

# ## <center> Target Marketing for Canadian Bank </center>
# 

# **Scenario**
# 
# In order to increase credit balances, Bank “A” is looking to execute a marketing campaign. The campaign will target existing clients and will offer them promotional interest rates to attract deposit balances.
# 
# **The Task**
# 
# You are tasked to find right set of customers who are most likely to respond to the campaign.
# 
# **Datasets**
# 
# Training data –Training dataset contains 64000 observations from previous campaigns. Actual responses (labels) are in the variable called ‘Target’ Testing data –contains 1480 observations with 36 variables and no labels
# 
# **Evaluation Metric : ROCAUC Score**
# 
# 
# **ATTRIBUTE**
# 
# •  customer_id: This is a unique identifier for each customer, which can be used to link the data across different tables or sources.
# 
# •  Balance: This is the amount of money that the customer owes to the bank on their credit card account. A higher balance means that the customer has more debt and may be more interested in lower interest rates.
# 
# •  PreviousCampaignResult: This is a categorical variable that indicates whether the customer responded positively or negatively to the last marketing campaign. A positive response means that the customer accepted the offer, while a negative response means that they rejected it or did not respond at all.
# 
# •  Product1...Product6: These are binary variables that indicate whether the customer owns a certain product from the bank, such as a savings account, a mortgage, a loan, etc. A value of 1 means that the customer owns the product, while a value of 0 means that they do not.
# 
# •  Transaction1...Transaction9: These are numerical variables that represent the amount of money that the customer spent or received in their last 9 transactions with the bank. A positive value means that the customer received money, while a negative value means that they spent money.
# 
# •  External Accounts 1...External Accounts 7: These are numerical variables that represent the number of external accounts that the customer has with other financial institutions, such as other banks, credit unions, insurance companies, etc. A higher number of external accounts may indicate that the customer is more likely to shop around for better deals or switch to another provider.
# 
# •  Activity Indicator: This is a numerical variable that represents the number of activities that the customer performed with the bank in a given period, such as using telebanking, visiting a branch, using an ATM, etc. A higher activity indicator may indicate that the customer is more engaged with the bank and more loyal to its services.
# 
# •  Regular Interaction Indicator: This is a categorical variable that represents how frequently the customer interacts with the bank on a rating scale from 1 (very low) to 5 (very high). A higher regular interaction indicator may indicate that the customer is more satisfied with the bank and more likely to respond to its offers.
# 
# •  CompetitiveRate1 ... CompetitiveRate7: These are numerical variables that represent the interest rates that the bank offered to the customer on different products, such as savings accounts, loans, mortgages, etc. These rates are meant to be competitive and attractive to the customer and may vary depending on their profile and preferences.
# 
# •  RateBefore: This is a numerical variable that represents the interest rate that the customer had on their products before the competitive rates were offered. This rate may be higher or lower than the competitive rates depending on the market conditions and the customer's bargaining power.
# 
# •  ReferenceRate: This is a numerical variable that represents the interest rate that the customer agreed to have on their products after negotiating with the bank based on the competitive rates. This rate may be equal to or different from the competitive rates depending on how successful the negotiation was.

# ## 1. Data Preparation

# In[1]:


# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Load dataset into notebook
train_df = pd.read_csv("C:\\Users\\ASUS\\Downloads\\train\\train.csv")

# This test dataset is for submission, no label results given.
test_df = pd.read_csv("C:\\Users\\ASUS\\Downloads\\train\\train.csv")


# In[3]:


# Have a look at dataset
train_df.tail()


# In[4]:


train_df.head()


# In[5]:


# General Information for training set
train_df.info()


# In[6]:


# Have a look at dataset
test_df.tail()


# In[7]:


# General Information for test set
test_df.info()


# ## 2. Exploration Data Analysis

# In[8]:


# Check the original shapes of both training and test data
train_df.shape, test_df.shape


# ### Check the balance of target classes

# In[9]:


import plotly.graph_objects as go

total_customers = len(train_df)
group0_customers = len(train_df[train_df['Target'] == 0])
group1_customers = len(train_df[train_df['Target'] == 1])

fig = go.Figure()

# Total Customers
fig.add_trace(go.Indicator(
    mode="number",
    value=total_customers,
    title="Total Customers",
    number={'font': {'size': 50}},
    domain={'x': [0, 0.25], 'y': [0, 1]}
))

# Retained Customers
fig.add_trace(go.Indicator(
    mode="number",
    value=group0_customers,
    title="Customers Group 0",
#    titlefont={'size': 12, 'font-weight': 'bold'},
    number={'font': {'size': 50}},
    domain={'x': [0.3, 0.55], 'y': [0, 1]}
))

# Churned Customers
fig.add_trace(go.Indicator(
    mode="number",
    value=group1_customers,
    title="Customers Group 1",
    number={'font': {'size': 50}},
    domain={'x': [0.6, 0.85], 'y': [0, 1]}
))


# <div class="alert alert-block alert-warning">
# <b>Observations: </b> 
# 
# - We have a balanced dataset with both customer groups 0 and 1 with 32.01K customers classified as Group 0 and 31.99K customers classified as Group 1
# </div>

# ### Check missing values

# In[10]:


# Checking missing values for all features
train_df.isnull().sum()


# <div class="alert alert-block alert-warning">
# <b>Observations: </b> 
# 
# - We have no missing values in this dataset, therefore we do not have to deal with missing values in this case
# </div>

# In[11]:


# Checking unique values for all features
unique = train_df.nunique().sort_values(ascending=True).reset_index().rename(columns = {'index': 'Feature', 0: 'Unique_Count'})
unique


# <div class="alert alert-block alert-warning">
# <b>Observations: </b> 
# 
# - Except from Customer_id, we can tell that there are a lot of features with the number of unique values less than 10 which can be considered as categorical features (Already been pre-processed) regardless their data types.
# - Those features with the number of unique values greater than 20 can be considered numeric features which we need to pay attention to.
# </div>

# ### Check duplicates

# In[12]:


# Check the duplicates in our train_df
print('The number of duplicated rows with ID column: ', train_df.duplicated().sum())


# <div class="alert alert-block alert-warning">
# <b>Observations: </b> 
# 
# - We don't have any duplicate in this training dataset.
# </div>

# In[13]:


# Explore the training data
train_df.hist(figsize = (20, 20), bins = 50, xlabelsize = 8, ylabelsize = 8);


# <div class="alert alert-block alert-warning">
# <b>Observations: </b> 
# 
# - We can see that all features regarding Products and External Accounts have only two values 0 and 1 indicating whether or not the customers have that product or external account.
# - Those transactions and balance features are right-skewed as the large amount of data points for these features are 0 values
# </div>

# ### Exploring data types and its count

# In[14]:


# Separate features into correct data types
categorical_features = unique[(unique['Unique_Count'] <10)]['Feature'].tolist()

high_cardinality_features = unique[(unique['Unique_Count'] >= 10) & (unique['Unique_Count'] <=20)]['Feature'].tolist()

numeric_features=unique[(unique['Unique_Count'] > 20)]['Feature'].tolist()


# In[15]:


# Find the number of features based on their data types
data_types = ['Categorical', 'High_cardinality', 'Numeric']
num_lst = [len(categorical_features), len(high_cardinality_features), len(numeric_features)]

# Plotting the graph showing the difference between data types
plot = sns.barplot(x = data_types, y = num_lst, palette = 'magma')

# Show value for each bar in the graph
plot.bar_label(plot.containers[0])

plt.title('Distribution of different data types over the dataset')
plt.xlabel('Data Types')
plt.ylabel('Number of Features')
plt.show()


# <div class="alert alert-block alert-warning">
# <b>Observations: </b> 
# 
# - Regarding the number of unique values for our features, we have categorical and numeric features in this training dataset. Obviously, these categorical features have been pre-processed before so there is no need to pre-process them. However, we need to look closer to those numeric features to see if we need to do any transformation or scaling.
# </div>

# In[16]:


# Checking statistcal summary
train_df.drop("Target", axis = 1).describe().T


# In[17]:


# Explore box plots for our features 
import math

df2 = train_df.copy()

num_cols_filtered = train_df.drop(["Customer_id","Target"], axis = 1).columns

num_plots = len(num_cols_filtered)
num_rows = math.ceil(num_plots / 3)

fig, ax = plt.subplots(num_rows, 3, figsize=(40, 36))
ax = ax.flatten()

for idx, c in enumerate(num_cols_filtered):
    ax[idx].set_title(c)
    sns.boxplot(x='Target', y=c, data=df2, ax=ax[idx])
    ax[idx].set_ylabel('')

for i in range(len(num_cols_filtered), len(ax)):
    ax[i].axis('off')

plt.tight_layout(pad=5, w_pad=3, h_pad=5)  # Increase padding between subplots
plt.show()


# <div class="alert alert-block alert-warning">
# <b>Observations: </b> 
# 
# - Based on the statistical summary and boxplots for all numerical features, we see that some of our features have outliers. Most of them are from the **transactions group, activity indicator and regular interaction indicator (belong to the numeric features group)**.
# </div>

# In[18]:


# Study the correlation between high-correlated features and our target column - populairty
corr_1 = train_df.drop('Customer_id', axis = 1).corr()

plt.figure(figsize=(40, 40))
sns.heatmap(corr_1, cmap='RdBu', vmin=-1, vmax=1, annot=True)
plt.title("Correlation between target columns and selected features")
plt.show()


# <div class="alert alert-block alert-warning">
# <b>Observations: </b> 
# 
# - Based on the heat map above, we observe that the majority of existing features is weakly correlated or not correlated with the target column.
# - There are several features we can consider further for our analysis, including Transaction 4, Product 1, Product2, Balance, transactions feature groups, Activity Indicator, and Regular Interaction Indicator.
# </div>

# ### Further EDA with each group of features

# In[19]:


# Split up those features in specific groups for further EDA
product_cols = ['Product1', 'Product2', 'Product3', 'Product4', 'Product5', 'Product6']

transac_cols = ['Transaction1', 'Transaction2', 'Transaction3', 'Transaction4',
                'Transaction5', 'Transaction6', 'Transaction7', 'Transaction8', 'Transaction9']

ext_acc_cols = ['ExternalAccount1', 'ExternalAccount2', 'ExternalAccount3', 'ExternalAccount4', 
                'ExternalAccount5', 'ExternalAccount6', 'ExternalAccount7']

competitiveRate_cols = ['CompetitiveRate1', 'CompetitiveRate2', 'CompetitiveRate3', 'CompetitiveRate4', 
                        'CompetitiveRate5', 'CompetitiveRate6', 'CompetitiveRate7']


# ### Additional New Features to support EDA as well as further data analysis

# In[20]:


# Total products per customer using in this bank
train_df['Total_products'] = train_df.loc[:, product_cols].sum(axis =1)


# In[21]:


# Total external accounts per customer outside of this bank
train_df['Total_ext_acc'] = train_df.loc[:, ext_acc_cols].sum(axis =1)


# In[22]:


# Total transactions per customer recently with this bank
train_df['No_Transactions'] = train_df[transac_cols].astype(bool).sum(axis = 1)


# #### Explore Total product for customer on different classes

# In[23]:


plt.figure(figsize=(12,6))
sns.countplot(data = train_df, x = 'Total_products', hue = 'Target', palette='Greens_r')
plt.title('Total Products over different target classes', fontsize=18, fontweight='bold')
plt.xlabel("Total Products")
plt.ylabel("Number of Customers")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(['Group 0', 'Group 1'], fontsize=16)


# <div class="alert alert-block alert-warning">
# <b>Observations: </b> 
# 
# - We conduct the sum of number of products that customers have been using from the bank, and we find that the majority of customers was not using any product from the bank on both customer target groups 0 and 1.
# - The rest of customers is using either 1 or more than 1 product from the bank and there is no clear difference between two target groups.
# </div>

# In[24]:


plt.figure(figsize=(12,6))
sns.countplot(data = train_df, x = 'Total_ext_acc', hue = 'Target', palette='Greens_r')
plt.title('Number of External Accounts over different target classes', fontsize=18, fontweight='bold')
plt.xlabel("Total External Accounts")
plt.ylabel("Number of Customers")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.legend(['Group 0', 'Group 1'], fontsize=16)


# <div class="alert alert-block alert-warning">
# <b>Observations: </b> 
# 
# - We conduct the sum of number of external accounts that customers have been using, and we find that the majority of customers has one external account and there is no clear difference between two customer target groups 0 and 1.
# </div>

# In[25]:


plt.figure(figsize=(12,6))
sns.countplot(data = train_df, x = 'RegularInteractionIndicator', hue = 'Target', palette='Greens_r')
plt.title('Regular Interaction Indicator over different target classes', fontsize=18, fontweight='bold')
plt.xlabel("Regular Interaction Indicator (Rating) ")
plt.ylabel("Number of Customers")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.legend(['Group 0', 'Group 1'], fontsize=16)


# <div class="alert alert-block alert-warning">
# <b>Observations: </b> 
# 
# - Regarding the Regular Interaction Indicator, it is obvious that the majority of customer has not been interacting with the bank
# </div>

# In[26]:


plt.figure(figsize=(12,6))
sns.countplot(data = train_df, x = 'No_Transactions', hue = 'Target', palette='Greens_r')
plt.title('Total of transactions over different target classes', fontsize=18, fontweight='bold')
plt.xlabel("Number of Recent Transactions ")
plt.ylabel("Number of Customers")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.legend(['Group 0', 'Group 1'], fontsize=16)


# <div class="alert alert-block alert-warning">
# <b>Observations: </b> 
# 
# - Target Group 1 customers tend to have more done transactions with bank, from at least 2 transactions over the last 9 given transactions
# - Target Group 0 customers don't make any transaction with the bank over the 9 given transactions or only one transaction (approxiamtely 12000 customers)
# </div>

# In[27]:


plt.figure(figsize=(12,6))
sns.countplot(data = train_df, x = 'RateBefore', hue = 'Target', palette='Greens_r')
plt.title('Interest Rate over different target classes', fontsize=18, fontweight='bold')
plt.xlabel("Interest Rate Before Negotiation ")
plt.ylabel("Number of Customers")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.legend(['Group 0', 'Group 1'], fontsize=16)


# <div class="alert alert-block alert-warning">
# <b>EDA Conclusion: </b> 
# 
# - At this stage of EDA, it is hard for us to see any clear distinct between two customer groups over those existing features. Therefore, it is impossible to get rid of any features as we do not want to lose any important insights hiding underneath those features.
#     
# - Finally, we decided to go further with building our model and feeding all features to observe the result.
# </div>

# ## 3. Building Prediction Model

# In[28]:


train_df.drop(['Total_products', 'Total_ext_acc', 'No_Transactions'], axis = 1, inplace = True)


# In[29]:


# sperate variables and prediction
y = train_df['Target']
X = train_df.drop(['Target', 'Customer_id'], axis=1, errors='ignore')


# In[30]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10086, stratify=y) #stratified sampling based on the target
print(f"X_train.shape: {x_train.shape}")
print(f"X_test.shape: {x_test.shape}")
print(f"y_train.shape: {y_train.shape}")
print(f"y_test.shape: {y_test.shape}")


# In[31]:


# Remove Customer_id from numerical features list
numeric_features.remove("Customer_id")


# In[32]:


#Create numerical pipeline to transform numerical values

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#Convert the non transformed Dataframe into list.
columns_categorical_list = categorical_features
columns_numerical_list = numeric_features

pipeline_numerical = Pipeline([
  ('scaler', StandardScaler()),
])

pipeline_full = ColumnTransformer([
  ("numerical", pipeline_numerical, columns_numerical_list),
])


# In[33]:


pipeline_full = ColumnTransformer([
    
    # Transform those numerical features
    ("numerical", pipeline_numerical, columns_numerical_list)],
    # Any other columns are ignored
    remainder="passthrough"
)

pipeline_full.fit(x_train)
x_train = pipeline_full.transform(x_train)
x_test = pipeline_full.transform(x_test)
print(f"X_train transformed.shape: {x_train.shape}")
print(f"X_test transformed.shape: {x_test.shape}")


# In[34]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

lr_model = LogisticRegression()

lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)
y_prob_lr = lr_model.predict(x_test)

print("Precision Score with LR: ", precision_score(y_test, y_prob_lr))
print("Recall Score with LR: ", recall_score(y_test, y_prob_lr))
print("F1 Score Score with LR: ", f1_score(y_test, y_prob_lr))
print("Accuracy Score with LR: ", accuracy_score(y_test, y_prob_lr))
print("ROC with Logistic Regression:", roc_auc_score(y_test, y_prob_lr))


# In[35]:


from sklearn.ensemble import RandomForestClassifier 

rfr_model = RandomForestClassifier()
rfr_model.fit(x_train,y_train)

y_prob_rfr = rfr_model.predict(x_test)

print("Precision Score with RandomForest: ", precision_score(y_test, y_prob_rfr))
print("Recall Score with RandomForest: ", recall_score(y_test, y_prob_rfr))
print("F1 Score Score with RandomForest: ", f1_score(y_test, y_prob_rfr))
print("Accuracy Score with RandomForest: ", accuracy_score(y_test, y_prob_rfr))
print("ROC by RandomForest:",roc_auc_score(y_test,y_prob_rfr))


# In[36]:


from sklearn.ensemble import AdaBoostClassifier

AB_model = AdaBoostClassifier(random_state=2022)
AB_model.fit(x_train,y_train)

y_prob_AB = AB_model.predict(x_test)

print("Precision Score with AdaBoost: ", precision_score(y_test, y_prob_AB))
print("Recall Score with AdaBoost: ", recall_score(y_test, y_prob_AB))
print("F1 Score Score with AdaBoost: ", f1_score(y_test, y_prob_AB))
print("Accuracy Score with AdaBoost: ", accuracy_score(y_test, y_prob_AB))
print("ROC by AdaBoost:",roc_auc_score(y_test,y_prob_AB))


# In[37]:


from sklearn.ensemble import GradientBoostingClassifier
GBDT_model = GradientBoostingClassifier(random_state=2022)
GBDT_model.fit(x_train,y_train)

y_prob_GBDT = GBDT_model.predict(x_test)

print("Precision Score with GradientBoosting: ", precision_score(y_test, y_prob_GBDT))
print("Recall Score with GradientBoosting: ", recall_score(y_test, y_prob_GBDT))
print("F1 Score Score with GradientBoosting: ", f1_score(y_test, y_prob_GBDT))
print("Accuracy Score with GradientBoosting: ", accuracy_score(y_test, y_prob_GBDT))
print("RUC by GradientBoosting",roc_auc_score(y_test,y_prob_GBDT))


# In[38]:


import xgboost as xgb
from sklearn.metrics import roc_auc_score

XGB_model = xgb.XGBClassifier()
XGB_model.fit(x_train, y_train)
y_prob_xgb = XGB_model.predict(x_test)

print("Precision Score with XGB: ", precision_score(y_test, y_prob_xgb))
print("Recall Score with XGB: ", recall_score(y_test, y_prob_xgb))
print("F1 Score Score with XGB: ", f1_score(y_test, y_prob_xgb))
print("Accuracy Score with XGB: ", accuracy_score(y_test, y_prob_xgb))
print("ROC by XGB:", roc_auc_score(y_test, y_prob_xgb))


# In[39]:


from tabulate import tabulate

roc_data = pd.DataFrame([['Logistic Regression', precision_score(y_test, y_prob_lr), recall_score(y_test, y_prob_lr), f1_score(y_test, y_prob_lr), accuracy_score(y_test, y_prob_lr),roc_auc_score(y_test, y_prob_lr)],
                         ['XGB', precision_score(y_test, y_prob_xgb), recall_score(y_test, y_prob_xgb), f1_score(y_test, y_prob_xgb), accuracy_score(y_test, y_prob_xgb),roc_auc_score(y_test, y_prob_xgb)],
                         ['Random Forest', precision_score(y_test, y_prob_rfr), recall_score(y_test, y_prob_rfr), f1_score(y_test, y_prob_rfr), accuracy_score(y_test, y_prob_rfr),roc_auc_score(y_test, y_prob_rfr)],
                         ['AdaBoost', precision_score(y_test, y_prob_AB), recall_score(y_test, y_prob_AB), f1_score(y_test, y_prob_AB), accuracy_score(y_test, y_prob_AB),roc_auc_score(y_test, y_prob_AB)],
                         ['GBDT', precision_score(y_test, y_prob_GBDT), recall_score(y_test, y_prob_GBDT), f1_score(y_test, y_prob_GBDT), accuracy_score(y_test, y_prob_GBDT),roc_auc_score(y_test, y_prob_GBDT)]],
                        columns=['Algorithm', 'Precision Score', 'Recall Score', 'F1 Score', 'Accuracy Score','ROC AUC Score'])

print(tabulate(roc_data, headers='keys', tablefmt='pretty', showindex=False))
roc_data.to_excel('Model_Performance.xlsx', index=False)


# In[40]:


def plot_importance(model, features, num=len(X)):
    plt.figure(figsize=(7, 7))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=features.sort_values(by="Value", ascending=False)[0:num])
    plt.title('Feature Importance for Random Forest Classifier')
    plt.tight_layout()
    plt.show()

features = X.columns
importances = rfr_model.feature_importances_

feat_importances = pd.DataFrame()
feat_importances['Feature'] = features
feat_importances['Value'] = importances

# Call the plot_importance function
plot_importance(GBDT_model, feat_importances)

feat_importances.sort_values('Value', ascending=False).to_excel('Logistic_Regression_Feature_Importance.xlsx')

# Print fancy table
sorted_feat_importances = feat_importances.sort_values('Value', ascending=False)
# print(tabulate(sorted_feat_importances, headers='keys', tablefmt='pretty', showindex=False))


# In[41]:


def plot_importance(model, features, num=len(X)):
    plt.figure(figsize=(7, 7))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=features.sort_values(by="Value", ascending=False)[0:num])
    plt.title('Feature Importance for AdaBoost Classifier')
    plt.tight_layout()
    plt.show()

features = X.columns
importances = AB_model.feature_importances_

feat_importances = pd.DataFrame()
feat_importances['Feature'] = features
feat_importances['Value'] = importances

# Call the plot_importance function
plot_importance(GBDT_model, feat_importances)

feat_importances.sort_values('Value', ascending=False).to_excel('Logistic_Regression_Feature_Importance.xlsx')

# Print fancy table
sorted_feat_importances = feat_importances.sort_values('Value', ascending=False)
# print(tabulate(sorted_feat_importances, headers='keys', tablefmt='pretty', showindex=False))


# In[42]:


def plot_importance(model, features, num=len(X)):
    plt.figure(figsize=(7, 7))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=features.sort_values(by="Value", ascending=False)[0:num])
    plt.title('Feature Importance for Gradient Boosting Classifier')
    plt.tight_layout()
    plt.show()

features = X.columns
importances = GBDT_model.feature_importances_

feat_importances = pd.DataFrame()
feat_importances['Feature'] = features
feat_importances['Value'] = importances

# Call the plot_importance function
plot_importance(GBDT_model, feat_importances)

feat_importances.sort_values('Value', ascending=False).to_excel('User_Churn_GradientBoost_Feature_Importance.xlsx')

# Print fancy table
sorted_feat_importances = feat_importances.sort_values('Value', ascending=False)
# print(tabulate(sorted_feat_importances, headers='keys', tablefmt='pretty', showindex=False))


# In[43]:


def plot_importance(model, features, num=len(X)):
    plt.figure(figsize=(7, 7))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=features.sort_values(by="Value", ascending=False)[0:num])
    plt.title('Feature Importance for XGB')
    plt.tight_layout()
    plt.show()

features = X.columns
importances = XGB_model.feature_importances_

feat_importances = pd.DataFrame()
feat_importances['Feature'] = features
feat_importances['Value'] = importances

# Call the plot_importance function
plot_importance(GBDT_model, feat_importances)

feat_importances.sort_values('Value', ascending=False).to_excel('User_Churn_GradientBoost_Feature_Importance.xlsx')

# Print fancy table
sorted_feat_importances = feat_importances.sort_values('Value', ascending=False)
# print(tabulate(sorted_feat_importances, headers='keys', tablefmt='pretty', showindex=False))


# In[44]:


#From the feature importance for Gradient Boosting Classifier above, we decided to choose these features:
selected_features = ['Balance','Transaction4', 'Transaction6', 'ActivityIndicator', 'Transaction2',
                     'Transaction5','Transaction8','Transaction3','Product1','Product2', 'Transaction7',
                     'Transaction9','RegularInteractionIndicator','Transaction1']


# In[45]:


from sklearn.model_selection import train_test_split

# sperate variables and prediction
y = train_df['Target']
x = train_df[selected_features]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1000)


# ### We start to use selected features on the top 3 models with best performances from the previous round and also do the hyperparameter tuning
# .

# In[46]:


x_train.shape, x_test.shape


# In[47]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score


# In[48]:


rf_random.best_params_


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(bootstrap= True,
                             max_depth=20,
                             max_features='auto',
                             min_samples_leaf= 1,
                             min_samples_split= 150,
                             n_estimators= 200)

rfc.fit(x_train, y_train)
y_pred_rfc = rfc.predict(x_test)

print("Precision Score with RandomForest: ", precision_score(y_test, y_pred_rfc))
print("Recall Score with RandomForest: ", recall_score(y_test, y_pred_rfc))
print("F1 Score Score with RandomForest: ", f1_score(y_test, y_pred_rfc))
print("Accuracy Score with RandomForest: ", accuracy_score(y_test, y_pred_rfc))
print("ROC by RandomForest:",roc_auc_score(y_test,y_pred_rfc))


# ### ADABoost

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier

parameters = {
    'n_estimators': range(10, 150, 10),
    'learning_rate': [0.075, 0.1, 0.15,0.25],
}

AB_model = AdaBoostClassifier()
grid_search = GridSearchCV(AB_model, parameters, scoring='roc_auc', cv=3, n_jobs=-1)
grid_search.fit(x_train, y_train)
grid_search.best_params_


# In[ ]:


# ADAboost Classifier
from sklearn.ensemble import AdaBoostClassifier
AB_model = AdaBoostClassifier(learning_rate=0.25, n_estimators= 140, random_state=2022)
AB_model.fit(x_train,y_train)

y_prob_AB = AB_model.predict(x_test)

print("Precision Score with AdaBoost: ", precision_score(y_test, y_prob_AB))
print("Recall Score with AdaBoost: ", recall_score(y_test, y_prob_AB))
print("F1 Score Score with AdaBoost: ", f1_score(y_test, y_prob_AB))
print("Accuracy Score with AdaBoost: ", accuracy_score(y_test, y_prob_AB))
print("ROC by AdaBoost:",roc_auc_score(y_test,y_prob_AB))


# ### Gradient Boosting Classifier

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

parameters = {
    "n_estimators":[50, 250, 500],
    "max_depth":[1, 3, 5, 7, 9],
    "learning_rate":[0.01, 0.1, 0.15]
}

GBDT_model = GradientBoostingClassifier()
grid_search = GridSearchCV(GBDT_model,parameters,cv=3, verbose = 3, scoring = 'roc_auc')
grid_search.fit(x_train,y_train)
grid_search.best_params_


# In[ ]:


grid_search.best_params_


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
GBDT_model = GradientBoostingClassifier(learning_rate= 0.01, max_depth= 5, n_estimators= 500,random_state=2022)
GBDT_model.fit(x_train,y_train)

y_prob_GBDT1 = GBDT_model.predict(x_test)

print("Precision Score with GradientBoosting: ", precision_score(y_test, y_prob_GBDT1))
print("Recall Score with GradientBoosting: ", recall_score(y_test, y_prob_GBDT1))
print("F1 Score Score with GradientBoosting: ", f1_score(y_test, y_prob_GBDT1))
print("Accuracy Score with GradientBoosting: ", accuracy_score(y_test, y_prob_GBDT1))
print("ROC by GradientBoosting:",roc_auc_score(y_test,y_prob_GBDT1))


# ### XGB 

# In[ ]:


from sklearn.model_selection import KFold
params = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01,0.05,0.1],
    'booster': ['gbtree', 'gblinear'],
    'gamma': [0, 0.5, 1],
    'reg_alpha': [0, 0.5, 1],
    'reg_lambda': [0.5, 1, 5],
    'base_score': [0.2, 0.5, 1]
}

gs2 = GridSearchCV(xgb_model, params, cv=KFold(n_splits=3), scoring='roc_auc', verbose = 3)
gs2.fit(x_train, y_train)
gs2.best_params_


# In[ ]:


import xgboost as xgb
xgb_model = xgb.XGBClassifier(base_score=0.5,
 booster = 'gbtree',
 gamma= 0.5,
 learning_rate= 0.05,
 n_estimators= 100,
 reg_alpha= 1,
 reg_lambda= 5, random_state=2022)
xgb_model.fit(x_train,y_train)

y_pred_xgb = xgb_model.predict(x_test)

print("Precision Score with XGB: ", precision_score(y_test, y_pred_xgb))
print("Recall Score with XGB: ", recall_score(y_test, y_pred_xgb))
print("F1 Score Score with XGB: ", f1_score(y_test, y_pred_xgb))
print("Accuracy Score with XGB: ", accuracy_score(y_test, y_pred_xgb))
print("RUC by XGB:",roc_auc_score(y_test,y_pred_xgb))


# ### Gradient Boosting Classifier - Selected Model

# In[ ]:


from sklearn.model_selection import KFold, cross_val_score

k = 3 # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

scores_accuracy = cross_val_score(GBDT_model, x, y, cv=kf, scoring='accuracy')
scores_roc_auc = cross_val_score(GBDT_model, x, y, cv=kf, scoring='roc_auc')

print(f'Accuracy scores for each fold: {scores_accuracy}')
print(f'Average accuracy: {scores_accuracy.mean():.4f}')
print(f'Standard deviation: {scores_accuracy.std():.4f}')
print('\n')

print(f'ROC AUC scores for each fold: {scores_roc_auc}')
print(f'Average ROC AUC: {scores_roc_auc.mean():.4f}')
print(f'Standard deviation: {scores_roc_auc.std():.4f}')


# In[ ]:


from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score

y_pred_gb_best = GBDT_model.predict(x_test)  # Compute the predicted class labels
print("Gradient Boosting algorithm ROC score:", roc_auc_score(y_test, y_pred_gb_best))

fpr, tpr, _ = roc_curve(y_test, y_pred_gb_best)
f1 = f1_score(y_test, y_pred_gb_best)

plt.figure(figsize=(9, 5))
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc_score(y_test, y_pred_gb_best):.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Gradient Boosting Classifier')
plt.legend(loc="lower right")


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm = confusion_matrix(y_test, y_pred_gb_best)

accuracy = accuracy_score(y_test, y_pred_gb_best)
percent_accuracy = accuracy * 100

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='YlGnBu', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix GB Best Model (Accuracy: {percent_accuracy:.2f}%)')
plt.show()


# ### Training and Validation Curve (Please figure out thís part, I just copied from Ashish's notebook)

# In[ ]:


# #draw the learning curve on the best estimator.

# svc_randomized = rnd_search_cv.best_estimator_

# train_errors, val_errors = [], []
# for m in range(100, len(X_train), 100):
#     svc_randomized.fit(X_train[:m], y_tr_b[:m]) # DO NOT DELETE
#     y_train_predict = svc_randomized.predict(X_train[:m])
#     y_val_predict = svc_randomized.predict(X_test[:m])
#     train_errors.append(mean_squared_error(y_train_predict, y_tr_b[:m]))
#     val_errors.append(mean_squared_error(y_val_predict, y_te_b[:m]))
    
# plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
# plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
# plt.show()


# ### Apply on test data

# In[ ]:


test_df[selected_features]


# ### Random Forest

# In[ ]:


rfc


# In[ ]:


rfc_result = test_df[['Customer_id']]
rfc_result['Target'] = rfc.predict(test_df[selected_features])
rfc_result.to_csv(r"C:\Users\HP\Desktop\Intro to AI Projects\target-marketing-for-canadian-bank-2023s-aml-1413\Final_result\result_rfc1.csv", index=False)


# ### AdaBoost Classifier

# In[ ]:


AB_model


# In[ ]:


ab_result = test_df[['Customer_id']]
ab_result['Target'] = AB_model.predict_proba(test_df[selected_features])[:,1]
ab_result.to_csv(r"C:\Users\HP\Desktop\Intro to AI Projects\target-marketing-for-canadian-bank-2023s-aml-1413\Final_result\result_ab1.csv", index=False)


# ### Gradient Boosting

# In[ ]:


GBDT_model


# In[ ]:


gb_result = test_df[['Customer_id']]
gb_result['Target'] = GBDT_model.predict_proba(test_df[selected_features])[:,1]
# gb_result.to_csv(r"C:\Users\HP\Desktop\Intro to AI Projects\target-marketing-for-canadian-bank-2023s-aml-1413\Final_result\result_gb1.csv", index=False)


# ### XGB

# In[ ]:


xgb_model


# In[ ]:


xgb_result = test_df[['Customer_id']]
xgb_result['Target'] = xgb_model.predict_proba(test_df[selected_features])[:,1]
xgb_result.to_csv(r"C:\Users\HP\Desktop\Intro to AI Projects\target-marketing-for-canadian-bank-2023s-aml-1413\Final_result\result_xgb1.csv", index=False)


# <div class="alert alert-block alert-warning">
# <b>Conclusions: </b> 
# 
# - We conclude that our best model at this stage is Gradient Boosting Classifier with learning_rate=0.01, max_depth=5, n_estimators=500.
#     
# - We have considered the result after uploading on Kaggle and it shows that the result with the original selected features from the train data was 67.557 which is slightly higher than the similar approach but with data processing for some skewed transaction features (67.259)
# </div>

# ## 5. Neural Network

# ### Approach 1

# In[ ]:


pip install tensorflow --ignore-installed


# In[ ]:


# Importing tensorflow and keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Softmax, Dropout
from tensorflow.keras import optimizers


# In[ ]:


Create_model = Sequential()

#Input Layer
Create_model.add(Dense(x_train.shape[1], activation='relu', input_dim = x_train.shape[1]))

#Hidden Layer
Create_model.add(Dense(512,kernel_initializer='normal', activation='relu'))
Create_model.add(Dense(512,kernel_initializer='normal', activation='relu'))
Create_model.add(Dense(256,kernel_initializer='normal', activation='relu'))
Create_model.add(Dense(128,kernel_initializer='normal', activation='relu'))
Create_model.add(Dense(64,kernel_initializer='normal', activation='relu'))
Create_model.add(Dense(32,kernel_initializer='normal', activation='relu'))
#Output Layer
Create_model.add(Dense(1,kernel_initializer='normal', activation = 'sigmoid'))


# In[ ]:


#Compile the network

Create_model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
Create_model.summary()


# In[ ]:


tf.keras.utils.plot_model(Create_model, show_shapes=True)


# In[ ]:


#library to use KerasClassifier
get_ipython().system('pip install scikeras')


# In[ ]:


from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor
from sklearn.metrics import make_scorer, f1_score


batch_size = [50,75]
epochs = [3,5]

Hyp_Model_1 = KerasRegressor(model=Create_model)

param_grid = dict(batch_size=batch_size, epochs = epochs)
randSearch_1 = GridSearchCV(Hyp_Model_1, param_grid, cv=3, scoring=make_scorer(f1_score), n_jobs=-1, verbose=0)


# In[ ]:


y_train.value_counts()


# In[ ]:


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="5"
history = randSearch_1.fit(x_train, y_train,verbose=0)


# In[ ]:


best_params = randSearch_1.best_params_
best_estimators = randSearch_1.best_estimator_

print(best_params)
print(best_estimators)


# In[ ]:


y_prediction = best_estimators.predict(x_test)


# In[ ]:


y_test.value_counts()


# In[ ]:


y_predicted=[1 if i >= 0.5 else 0 for i in y_prediction]
print(y_predicted[:10])


# In[ ]:


from sklearn.metrics import mean_squared_error
import numpy as np

print("Precision Score with ANN: ", precision_score(y_test, y_predicted))
print("Recall Score with ANN: ", recall_score(y_test, y_predicted))
print("F1 Score Score with ANN: ", f1_score(y_test, y_predicted))
print("Accuracy Score with ANN: ", accuracy_score(y_test, y_predicted))
print("ROC by ANN:", roc_auc_score(y_test, y_predicted))


# ### Approach 2: Another way to Train Keras Models with L2 and Dropouts

# In[ ]:


Create_model_2 = Sequential()

#Input Layer
Create_model_2.add(Dense(x_train.shape[1], activation='relu', input_dim = x_train.shape[1]))

#Hidden Layer
Create_model_2.add(Dense(512,kernel_initializer='normal', activation='relu', kernel_regularizer=tf.keras.regularizers.L1(0.01), bias_regularizer=tf.keras.regularizers.l2(0.015)))
Create_model_2.add(Dropout(0.3))
Create_model_2.add(Dense(256,kernel_initializer='normal', activation='relu'))
Create_model_2.add(Dropout(0.3))
Create_model_2.add(Dense(128,kernel_initializer='normal', activation='relu', kernel_regularizer=tf.keras.regularizers.L1(0.01), bias_regularizer=tf.keras.regularizers.l2(0.015)))
Create_model_2.add(Dropout(0.3))
Create_model_2.add(Dense(64,kernel_initializer='normal', activation='relu'))
Create_model_2.add(Dropout(0.3))
Create_model_2.add(Dense(32,kernel_initializer='normal', activation='relu'))
Create_model_2.add(Dropout(0.3))
#Output Layer
Create_model_2.add(Dense(1,kernel_initializer='normal', activation = 'sigmoid'))

opt = keras.optimizers.Adam(learning_rate=0.01)

Create_model_2.compile(loss = 'binary_crossentropy', optimizer=opt, metrics=['accuracy'])
Create_model_2.summary()


# #### Early Stopping using Keras Callback

# In[ ]:


from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode = 'min',verbose=1, patience=5)

history = Create_model_2.fit(x_train, y_train, batch_size=25, epochs=20, callbacks=[es])


# In[ ]:


from matplotlib import pyplot

train_acc = Create_model_2.evaluate(x_train, y_train)
test_acc = Create_model_2.evaluate(x_test, y_test)

y_prediction = Create_model_2.predict(x_test)

from sklearn.metrics import mean_squared_error
import numpy as np

print("Precision Score with ANN: ", precision_score(y_test, y_predicted))
print("Recall Score with ANN: ", recall_score(y_test, y_predicted))
print("F1 Score Score with ANN: ", f1_score(y_test, y_predicted))
print("Accuracy Score with ANN: ", accuracy_score(y_test, y_predicted))
print("ROC by ANN:", roc_auc_score(y_test, y_prediction))


# ## Comparison of results for  ANN models

# In[1]:


from tabulate import tabulate

# Define the data
data = [
    ["First ANN Model", 0.6422202001819837, 0.5467079783113865, 0.5906276150627615, 0.6178125, 0.6184288512518319],
    ["Second ANN Model", 0.0, 0.0, 0.0, 0.495703125, 0.5],
    ["Third ANN Model", 0.6355839663735526, 0.5930669800235018, 0.6131623119510521, 0.6274145081001563, 0.6273218738895763]
]

# Define the headers
headers = ["Model", "Precision", "Recall", "F1 Score", "Accuracy", "ROC"]

# Tabulate the data
table = tabulate(data, headers, tablefmt="grid")

# Print the table
print(table)


# >First ANN Model:
# Achieved moderate precision (0.642) and recall (0.547).
# F1 score (0.591) indicates a reasonable balance between precision and recall.
# Good accuracy (0.618), but may need further optimization to improve.
# 
# >Second ANN Model:
# High recall (1.000), capturing all true positives, but extremely low precision (0.504).
# F1 score (0.670) suggests a trade-off between recall and precision.
# Low accuracy (0.504) due to high false positive rate.
# 
# >Third ANN Model:
# Achieved a good balance between precision (0.640) and recall (0.580).
# F1 score (0.608) indicates a reasonably balanced performance.
# High accuracy (0.624) suggests overall good predictive ability.
# Decent ROC AUC (0.624), indicating good discrimination between classes.
# 
# Based on this comparison, the Third ANN Model stands out as the most balanced choice, with relatively good precision, recall, F1 score, accuracy, and ROC AUC.

# In[ ]:


Comparing results of statistical


# --------------
# Code starts here
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
from sklearn.preprocessing import Imputer
import numpy as np
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder
#### Data 1
# Load the data
df=pd.read_csv(path)


# Overview of the data
df.info()
df.describe()

# Histogram showing distribution of car prices
plt.figure(figsize=(10,10))
sns.distplot(df['price'],kde=True,rug=True)

# Countplot of the make column
plt.figure(figsize=(10,10))
sns.countplot(y='make',data=df)

# Jointplot showing relationship between 'horsepower' and 'price' of the car
plt.figure(figsize=(10,10))
sns.jointplot(x="horsepower", y="price", data=df,kind='reg')

# Correlation heat map
plt.figure(figsize=(15,15,))
sns.heatmap(data=df.corr(),cmap="YlGnBu")

# boxplot that shows the variability of each 'body-style' with respect to the 'price'
plt.figure(figsize=(12,8))
sns.boxplot(x="body-style",y="price",data=df)
#### Data 2

# Load the data
df1=pd.read_csv(path2)
df1.head(n=10)
# Impute missing values with mean
df1=df1.replace("?",np.NaN)
df1.isna().sum()

mean_imputer=Imputer(missing_values="NaN",strategy="mean")
df1[['normalized-losses']]=mean_imputer.fit_transform(df1[['normalized-losses']])
df1[['horsepower']]=mean_imputer.fit_transform(df1[['horsepower']])
# Skewness of numeric features
encoder = LabelEncoder()
numeric_columns=df1._get_numeric_data().columns
for i in numeric_columns:
    if skew(df1[i]) > 1:
        df1[i]=np.sqrt(df1[i])



# Label encode 

# Initialize encoder object
encoder = LabelEncoder()
#print(df.dtypes)
categorical_cols=df1.select_dtypes(include='object').columns

for i in categorical_cols:
    df1[i]=encoder.fit_transform(df1[i])
print(df1[categorical_cols].head(n=5))

#Combine the 'height' and 'width' to make a new feature 'area' of the frame of the car.
# Code ends here

df1['area']=df['height']*df1['width']
print(df1['area'][:5])



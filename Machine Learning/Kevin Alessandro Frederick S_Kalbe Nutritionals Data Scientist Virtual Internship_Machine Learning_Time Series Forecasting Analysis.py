#!/usr/bin/env python
# coding: utf-8

# # Data and Library

# In[1]:


#pip install pmdarima


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pmdarima as pm

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from itertools import permutations
from datetime import datetime
import pytz


# In[3]:


#Read all csv files
customer = pd.read_csv("C:\Kuliah\INTERNSHIPS\Kalbe Nutritionals Internship\Final Project\Data\Customer.csv")
transaction = pd.read_csv("C:\Kuliah\INTERNSHIPS\Kalbe Nutritionals Internship\Final Project\Data\Transaction.csv")
product = pd.read_csv("C:\Kuliah\INTERNSHIPS\Kalbe Nutritionals Internship\Final Project\Data\Product.csv")
store = pd.read_csv("C:\Kuliah\INTERNSHIPS\Kalbe Nutritionals Internship\Final Project\Data\Store.csv")


# In[4]:


customer.head()


# In[5]:


transaction.head()


# In[6]:


product.head()


# In[7]:


store.head()


# In[8]:


# Merge all data into a united new dataframe
df = pd.merge(transaction,customer,on='CustomerID')
df = pd.merge(df,product,on='ProductID', suffixes=('_Customer','_Product'))
df = pd.merge(df,store,on='StoreID')
df.head()


# # Data Cleaning

# # Check Data Type

# In[9]:


df.dtypes


# Kita akan mengubah data type variabel "Date" menjadi datetime dan variabel "Income" menjadi float dengan pemisah desimal berupa "."

# In[10]:


#Convert date and income data type
df['Date'] = pd.to_datetime(df['Date'])
df['Income'] = df['Income'].map(lambda x: float(x.replace(',','.')))

df.head()


# # Drop Irrelevant Columns

# In[11]:


# Drop irrelevant columns
df = df.drop(columns=['Latitude','Longitude'])
df.head()


# # Check Missing Value

# In[12]:


df.isna().sum()


# Variabel Marital Status memiliki missing value, maka akan dilakukan imputasi terhadap missing value menggunakan K-Nearest Neighbour Method

# In[13]:


from sklearn.preprocessing import LabelEncoder

# Create an instance of the LabelEncoder class
le = LabelEncoder()

# Fit the encoder to the Marital Status variable and transform it
df['Marital Status'] = le.fit_transform(df['Marital Status'])

# Print the encoded values
df.head()


# In[14]:


#Impute nan values using KNNImputer
#Lets use customer data to support imputer process
df_impute = df[['Age','Gender','Income','Marital Status']]

imputer = KNNImputer(n_neighbors=2)
df_impute = imputer.fit_transform(df_impute)
df_impute = pd.DataFrame(data=df_impute,columns=['Age','Gender','Income','Marital Status'])

print('Missing Value :',df_impute.isna().sum())


# In[15]:


df['Marital Status'] = df_impute["Marital Status"].astype('int')
df.isna().sum()


# In[16]:


df.dtypes


# # Export the Merged Cleaned Data

# In[17]:


# Export the cleaned data into a csv format
df.to_csv('C:\Kuliah\INTERNSHIPS\Kalbe Nutritionals Internship\Final Project\Data\Merged_Data.csv', index=False)


# # Time Series Analysis (Machine Learning)

# In[18]:


df_tsa = df.groupby('Date')[['Qty']].sum()
df_tsa


# In[19]:


df_tsa.plot()


# # Training and Testing Data Split

# In[20]:


#Split train and test
df_train = df_tsa.iloc[:-31]
df_test = df_tsa.iloc[-31:]


# # Pengecekan Stasioneritas Data

# Hipotesis uji stasioner data deret waktu menggunakan Augmented Dickey-Fuller (ADF) adalah:
# 
# H0 : data tidak stasioner </br>
# H1 : data stasioner

# In[21]:


from statsmodels.tsa.stattools import adfuller
adf_test = adfuller(df_train)
print(f'p-value: {adf_test[1]}')


# Nilai P-Value = 2.44 x 10^-30 < alpha = 0.05 sehingga tolak H0. Maka, diperoleh kesimpulan bahwa data stasioner sehingga pemodelan ARIMA dapat dilanjutkan.

# In[22]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

acf_original = plot_acf(df_train)

pacf_original = plot_pacf(df_train)


# Seperti yang terlihat pada gambar di atas, plot ACF dan PACF memiliki pola cuts off. Pada plot ACF lag yang signifikan adalah lag 1, dan pada plot PACF lag yang signifikan juga merupakan lag 1. Sehingga dapat diketahui model dugaannya adalah ARIMA
# (1, 0, 0) dan ARIMA (0, 0, 1).

# In[23]:


from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(df_train, order=(1,0,0))
model_fit = model.fit()
print(model_fit.summary())


# # Parameter Tuning Data Training

# In[24]:


#Manual parameter tuning
def tune(z,y,x):
    model = ARIMA(df_train, order=(x,y,z))
    model_fit = model.fit()
    forecast_test = model_fit.forecast(len(df_test))
    df_plot = df_tsa[['Qty']].iloc[-61:]

    df_plot['forecast'] = [None]*(len(df_plot)-len(forecast_test)) + list(forecast_test)
    
    MAE = mean_absolute_error(df_test, forecast_test)
    MAPE = mean_absolute_percentage_error(df_test, forecast_test)
    RMSE = np.sqrt(mean_squared_error(df_test, forecast_test))
    
    return MAE,MAPE,RMSE
    
#Parameter combinations
p = [1,0]
d = [0]
q = [0,1]

comb = []
for i in p:
    for j in d:
        for k in q:
            comb.append((i,j,k))

parameter = []
MAE_score = []
MAPE_score = []
RMSE_score = []

for i in comb:
    parameter.append(i)
    score = tune(*i)
    MAE_score.append(score[0])
    MAPE_score.append(score[1])
    RMSE_score.append(score[2])
    
tuning_df = pd.DataFrame({'Parameter':parameter,'MAE':MAE_score,'MAPE':MAPE_score,'RMSE':RMSE_score})
tuning_df.sort_values(by='MAE').head()


# Diputuskan untuk menggunakan parameter pdq ARIMA (1,0,1) karena memiliki nilai MAPE terkecil

# In[25]:


import matplotlib.pyplot as plt
residuals = model_fit.resid[1:]
fig, ax = plt.subplots(1,2)
residuals.plot(title='Residuals', ax=ax[0])
residuals.plot(title='Density', kind='kde', ax=ax[1])
plt.show()


# In[26]:


#Manual parameter tuning
model = ARIMA(df_train, order=(1, 0, 1))
model_fit = model.fit()


# In[27]:


#plot forecasting
forecast_test = model_fit.forecast(len(df_test))

df_plot = df_tsa[['Qty']].iloc[-61:]

df_plot['forecast_test'] = [None]*(len(df_plot)-len(forecast_test)) + list(forecast_test)

df_plot.plot()
plt.show()


# # Overall Quantity of Product Sold Forecasting

# In[28]:


#Overall Quantity Forecasting
model = ARIMA(df_tsa, order=(1, 0, 1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=31)


# In[29]:


#Plot forecasting
plt.figure(figsize=(12,5))
plt.plot(df_train, label='Training Data')
plt.plot(df_test, color='green', label='Test Data')
plt.plot(forecast,color='red', label= 'Quantity Forecast')
plt.title('Quantity Sold Forecasting')
plt.legend()
plt.show()


# In[30]:


forecast.mean()


# Berdasarkan hasil forecast di atas, diketahui bahwa estimasi kuantitas penjualan harian pada bulan januari 2023 adalah sekitar 51 pcs produk per hari (50.1262 dibulatkan ke atas).

# # Quantity of Each Product Forecast

# In[31]:


#Forecasting the quantity of each product for the next 31 days (January 2023 have 31 days in 1 month)
product_name = df['Product Name'].unique()

dfprod = pd.DataFrame({'Date':pd.date_range(start='2023-01-01',end='2023-01-31')})
dfprod = dfprod.set_index('Date')
for i in product_name:
    df1 = df[['Date','Product Name','Qty']]
    df1 = df1[df1['Product Name']==i]
    df1 = df1.groupby('Date')[['Qty']].sum()
    df1 = df1.reset_index()
    df_prod = pd.DataFrame({'Date':pd.date_range(start='2022-01-01',end='2022-12-31')})
    df_prod = df_prod.merge(df1, how='left', on='Date')
    df_prod = df_prod.fillna(0)
    df_prod = df_prod.set_index('Date')
    
    model1 = ARIMA(df_prod, order=(1,0,1))
    model1_fit = model1.fit()
    forecast1 = model1_fit.forecast(steps=31)
    dfprod[i] = forecast1.values
    
dfprod.head()


# In[32]:


#Forecasting Plot
plt.figure(figsize=(12,5))
plt.plot(dfprod)
plt.legend(dfprod.columns,loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('Quantity of Each Product Sold Forecast')
plt.show()


# In[33]:


#Plot forecasting
plt.figure(figsize=(12,5))
plt.plot(df_prod)
plt.plot(dfprod, label= 'Quantity of Each Product Forecast')
plt.title('Quantity of Each Product Sold Forecast')
plt.legend(dfprod.columns, loc='center left', bbox_to_anchor=(1,0.5))
plt.show()


# In[34]:


#Quantity of Each Product Sold forecast
round(dfprod.describe().T['mean'],0)


# Dari forecasting terhadap tiap produk yang terjual, diperkirakan pada bulan depan rata-rata produk Crackers akan terjual sebanyak 5 pcs per hari, Oat terjual sebanyak 3 pcs per hari, Thai Tea sebanyak 8 pcs per hari, Choco Bar sebanyak 7 pcs per hari, Coffee Candy sebanyak 6 pcs per hari, Yoghurt sebanyak 5 pcs per hari, Ginger Candy sebanyak 7 pcs per hari, Cheese Stick sebanyak 5 pcs per hari, Cashew sebanyak 2 pcs per hari, dan Potato Chip sebanyak 3 pcs per hari. Informasi ini dapat digunakan sebagai insight terhadap tim inventory untuk membuat stock persediaan harian yang cukup dan efektif.

# In[ ]:





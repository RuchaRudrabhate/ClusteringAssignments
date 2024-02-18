# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 11:20:39 2023

@author: arudr

"""

#Dataset -  EAST WEST AIRLINES.XLS

'''
The file EastWestAirlines contains information on passengers 
who belong to an airline’s frequent flier program.
For each passenger the data include information on their mileage 
history and on different ways they spent miles in the last year. 
'''
#Business Objective - 
'''
is to find clusters of passengers that are of the similar characteristics
to provide milage offers based on clustering /  groups

also to perform kmeans clustering to get optimal cluster number

minimize = cost
maximize -  offeres
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#IMPORT DATA SET AND CREATE DATAFRAME
airlines = pd.read_excel('C:/2-dataset/EastWestAirlines.xlsx')

airlines.dtypes
'''ID#               int64 Quantitative
Balance              int64 quantitative discrete
Qual_miles           int64 quantitative discrete
cc1_miles            int64 quantitative discrete
cc2_miles            int64 quantitative discrete
cc3_miles            int64 quantitative discrete
Bonus_miles          int64 quantitative discrete
Bonus_trans          int64 quantitative discrete
Flight_miles_12mo    int64 quantitative discrete
Flight_trans_12      int64 quantitative discrete
Days_since_enroll    int64 quantitative discrete
Award?               int64 nominal qualitative
dtype: object

In this data set all the columns are the integer type 
'''

cols = airlines.columns.values
airlines.shape
#(3999, 12)

airlines.describe()
desc = []
#describe each col
for i in cols:
    desc.append(airlines[i].describe())
    #print(airlines[i].describe())
    #print() 
print(desc)

'''
cols                min         max         mean 
ID                  1.00        4021.0      2014.81
Balance             0.000e+05   1.7048      7.3601
Qual_miles          0.000       11148.0     144.11
cc1_miles           1.00        5.000       2.0595
cc2_miles           1.000       3.000       1.0145
cc3_miles           1.000       5.0000      1.012253
Bonus_miles         0.000       263685.0    17144.8462
Bonus_trans         0.00        86.00       11.60
Flight_miles_12mo   0.00        30817.0     460.055
Flight_trans_12     0           53          1.3735
Days_since_enroll   2.00        8296.00     4118.559
Award?              0.00        1.00        0.37

**from this we can see that there is huge difference in min max and mean 
values of columns so this data need to normalised
      
'''

#initially we will perform EDA to analyse the data

#pairplot
import seaborn as sns
plt.close();
sns.set_style("whitegrid");
sns.pairplot(airlines, hue="Award?", height=3);
plt.show()

#pdf and cdf

counts, bin_edges = np.histogram(airlines['Balance'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)


#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='pdf')
'''
from pdf we can say that approx 90% of data have balance 20000
'''
plt.plot(bin_edges[1:], cdf,label = 'cdf')
#plt.hist(bin_edges[1:], cdf,label = 'cdf')
plt.legend()
plt.show();

#Boxplot and outliers treatment

sns.boxplot(airlines['Balance'])
sns.boxplot(airlines['Qual_miles'])
sns.boxplot(airlines['cc1_miles'])
sns.boxplot(airlines['cc2_miles'])
sns.boxplot(airlines['cc3_miles'])
sns.boxplot(airlines['Bonus_miles'])
sns.boxplot(airlines['Bonus_trans'])
sns.boxplot(airlines['Flight_miles_12mo'])
sns.boxplot(airlines['Flight_trans_12'])
sns.boxplot(airlines['Days_since_enroll'])
sns.boxplot(airlines['Award?'])

'''
from box plot except cc2 miles, days since enroll and award? 
all other colmns have outliers
we need to remove them
'''
#1
iqr = airlines['Balance'].quantile(0.75)-airlines['Balance'].quantile(0.25)
iqr
q1=airlines['Balance'].quantile(0.25)
q3=airlines['Balance'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
airlines['Balance'] =  np.where(airlines['Balance']>u_limit,u_limit,np.where(airlines['Balance']<l_limit,l_limit,airlines['Balance']))
sns.boxplot(airlines['Balance'])

#2
iqr = airlines['Qual_miles'].quantile(0.75)-airlines['Qual_miles'].quantile(0.25)
iqr
q1=airlines['Qual_miles'].quantile(0.25)
q3=airlines['Qual_miles'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
airlines['Qual_miles'] =  np.where(airlines['Qual_miles']>u_limit,u_limit,np.where(airlines['Qual_miles']<l_limit,l_limit,airlines['Qual_miles']))
sns.boxplot(airlines['Qual_miles'])

#3
iqr = airlines['cc1_miles'].quantile(0.75)-airlines['cc1_miles'].quantile(0.25)
iqr
q1=airlines['cc1_miles'].quantile(0.25)
q3=airlines['cc1_miles'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
airlines['cc1_miles'] =  np.where(airlines['cc1_miles']>u_limit,u_limit,np.where(airlines['cc1_miles']<l_limit,l_limit,airlines['cc1_miles']))
sns.boxplot(airlines['cc1_miles'])

#4
iqr = airlines['cc3_miles'].quantile(0.75)-airlines['cc3_miles'].quantile(0.25)
iqr
q1=airlines['cc3_miles'].quantile(0.25)
q3=airlines['cc3_miles'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
airlines['cc3_miles'] =  np.where(airlines['cc3_miles']>u_limit,u_limit,np.where(airlines['Bonus_miles']<l_limit,l_limit,airlines['Bonus_miles']))
sns.boxplot(airlines['cc3_miles'])

#5
iqr = airlines['Bonus_miles'].quantile(0.75)-airlines['Bonus_miles'].quantile(0.25)
iqr
q1=airlines['Bonus_miles'].quantile(0.25)
q3=airlines['Bonus_miles'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
airlines['Bonus_miles'] =  np.where(airlines['Bonus_miles']>u_limit,u_limit,np.where(airlines['Bonus_miles']<l_limit,l_limit,airlines['Bonus_miles']))
sns.boxplot(airlines['Bonus_miles'])

#6
iqr = airlines['Bonus_trans'].quantile(0.75)-airlines['Bonus_trans'].quantile(0.25)
iqr
q1=airlines['Bonus_trans'].quantile(0.25)
q3=airlines['Bonus_trans'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
airlines['Bonus_trans'] =  np.where(airlines['Bonus_trans']>u_limit,u_limit,np.where(airlines['Bonus_trans']<l_limit,l_limit,airlines['Bonus_trans']))
sns.boxplot(airlines['Bonus_trans'])

#7
iqr = airlines['Flight_miles_12mo'].quantile(0.75)-airlines['Flight_miles_12mo'].quantile(0.25)
iqr
q1=airlines['Flight_miles_12mo'].quantile(0.25)
q3=airlines['Flight_miles_12mo'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
airlines['Flight_miles_12mo'] =  np.where(airlines['Flight_miles_12mo']>u_limit,u_limit,np.where(airlines['Flight_miles_12mo']<l_limit,l_limit,airlines['Flight_miles_12mo']))
sns.boxplot(airlines['Flight_miles_12mo'])

#8
iqr = airlines['Flight_trans_12'].quantile(0.75)-airlines['Flight_trans_12'].quantile(0.25)
iqr
q1=airlines['Flight_trans_12'].quantile(0.25)
q3=airlines['Flight_trans_12'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
airlines['Flight_trans_12'] =  np.where(airlines['Flight_trans_12']>u_limit,u_limit,np.where(airlines['Flight_trans_12']<l_limit,l_limit,airlines['Flight_trans_12']))
sns.boxplot(airlines['Flight_trans_12'])

#now describe dataset
airlines.describe()
#we can see that there is huge difference between min,max and mean
# values for all the columns so we need to normalize the dataset

#initially normalize the dataset
def norm_fun(i):
    x = (i-i.min())/(i.max()-i.min())
    return x

#apply this func on airlines dataset
df_normal = norm_fun(airlines)
b = df_normal.describe()
b
#as qual miles is containing NAN values so we will drop it
df_normal.drop(['Qual_miles'],axis=1,inplace=True)

#now all the data is normalized
#dendrogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z = linkage(df_normal,method='complete',metric='euclidean')
plt.figure(figsize=(15,8))
plt.title('Hierarchical clustering dendrogram')
plt.xlabel('index')
plt.ylabel('distance')
#dendrogram
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()
#now apply clustering 
from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters=3,
                                     linkage='complete',
                                     affinity='euclidean').fit(df_normal)
#apply labels to clusters
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
#assign this series to autoIns dataframe as column
airlines['cluster'] = cluster_labels

airlinesNew = airlines.iloc[:,[-1,0,1,2,3,4,5,6,7,8,9,10,11]]
airlinesNew.iloc[:,2:].groupby(airlinesNew.cluster).mean()

airlinesNew.to_csv("AirlinesNew.csv",encoding='utf-8')
airlinesNew.cluster.value_counts()
import os
os.getcwd()


#############################################
#KMeans Clustering on east west airlines
#for this we will used normalized data set df_normal

from sklearn.cluster import KMeans
#total sum of squares
TWSS = []

#initially we will find the ideal cluster number using elbow curve

k = list(range(2,8))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_normal)
    TWSS.append(kmeans.inertia_)
  
TWSS
'''
[2908.260739348439,
 2263.316171028324,
 1932.897637296432,
 1667.4170613324961,
 1484.8283001092923,
 1313.3078600029346]
'''

'''
k selected by calculating the difference or decrease in
twss value 
'''
def find_cluster_number(TWSS):
    diff =[]
    for i in range(0,len(TWSS)-1):
        d = TWSS[i]-TWSS[i+1]
        diff.append(d)
    max = 0
    k =0
    for i in range(0,len(diff)):
        if max<diff[i]:
            max = diff[i]
            k = i+3
    return k

k = find_cluster_number(TWSS)
print("Cluster number is = ",k)
plt.plot(k,TWSS,'ro-')
plt.xlabel('No of clusters')
plt.ylabel('Total_within_SS')

model = KMeans(n_clusters=k)
model.fit(df_normal)
model.labels_
mb = pd.Series(model.labels_)
df_normal['clusters'] = mb
df_normal.head()
df_normal.shape
df_normal.columns
df_normal = df_normal.iloc[:,[-1,0,1,2,3,4,5,6,7,8,9,10]]
df_normal
df_normal.iloc[:,2:11].groupby(df_normal.clusters).mean()
df_normal.to_csv('C:/4-data mining/k_means_EastWestAirlines.csv')
import os
os.getcwd()


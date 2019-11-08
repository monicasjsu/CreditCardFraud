import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import sklearn.preprocessing

df = pd.read_csv('creditcard.csv')
#print(df.head())
#print(df.describe())

#print(df.isnull().sum())
print("The count of each class in raw,highly skewed dataset")
print(df['Class'].value_counts())

# sns.countplot(x='Class', data=df)
# plt.show()

# sns.pairplot(df_down_sampled)
# plt.show()

std_scaler = sklearn.preprocessing.StandardScaler()
rob_scaler = sklearn.preprocessing.RobustScaler()

df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1, 1))

df.drop(['Time', 'Amount'], axis=1, inplace=True)

scaled_amount = df['scaled_amount']
scaled_time = df['scaled_time']

df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
df.insert(0, 'scaled_amount', scaled_amount)
df.insert(1, 'scaled_time', scaled_time)
print("After performing the robust sampling on df, the data set is:")
print(df.head())
# df = df.sample(frac=1)

# amount of fraud classes 492 rows.
print("After downsampling the majority not-fraud column(0):")
fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0][:492]

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

# Shuffle dataframe rows
new_df = normal_distributed_df.sample(frac=1, random_state=42)

new_df.head()
print(new_df['Class'].value_counts())



# f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 20))
# corr = df.corr()
# sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size': 20}, ax=ax1)
# ax1.set_title("Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=14)
# plt.show()
print("After applying stratifying sampling, the Fraud(1) and non-fraud(0) values are:")
train, test = train_test_split(df, test_size=0.30, random_state=100, stratify=df['Class'])
train_df = pd.DataFrame(train)
test_df = pd.DataFrame(test)

train_df.sample()

print(train_df['Class'].value_counts())

df_majority = train_df[train_df.Class == 0]
df_minority = train_df[train_df.Class == 1]

majority_count = df_majority['Class'].count()
minority_count = df_minority['Class'].count()

majority_down_sampled = resample(df_majority, replace=False, n_samples=minority_count, random_state=100)
df_majority_down_sampled = pd.DataFrame(majority_down_sampled)

df_down_sampled = pd.concat([df_majority_down_sampled, df_minority])

print(df_down_sampled['Class'].value_counts())

#sns.countplot('Class', data=df_down_sampled)
#plt.title('Equally Distributed Classes', fontsize=14)
#plt.show()


#For creating a correlation matrix for downsampled data
#
#f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24,20))
#geatmap for just the downsampled data
#sub_sample_corr = df_down_sampled.corr()
#sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax2)
#ax2.set_title('SubSample Correlation Matrix \n (use for reference)', fontsize=14)
#plt.show()
#


print("From the heatmap plot;observations are:\n Positively correlated features are: V2,V4,V11,V19\nNegatively correlated features are: V16,V14,V12,V10,V3")
print("constructing box plots for the positively correlated features: V2,V4,V11,V19\n")
f, axes = plt.subplots(ncols=5, figsize=(20,5))

# Positive correlations (The higher the feature the probability increases that it will be a fraud transaction)
sns.boxplot(x="Class", y="V2", data=df_down_sampled, ax=axes[0])
axes[0].set_title('V2 vs Class Positive Correlation')

sns.boxplot(x="Class", y="V4", data=df_down_sampled, ax=axes[1])
axes[1].set_title('V4 vs Class Positive Correlation')


sns.boxplot(x="Class", y="V11", data=df_down_sampled, ax=axes[2])
axes[2].set_title('V11 vs Class Positive Correlation')


sns.boxplot(x="Class", y="V11", data=df_down_sampled, ax=axes[3])
axes[3].set_title('V11 vs Class Positive Correlation')

sns.boxplot(x="Class", y="V19", data=df_down_sampled, ax=axes[4])
axes[4].set_title('V19 vs Class Positive Correlation')

plt.show()

print("constructing box plots for the positively correlated features: V2,V4,V11,V19\n")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler

df = pd.read_csv('creditcard.csv')
print(df.head())
print(df.describe())
print("Null values in the given the dataset:")
print(df.isnull().sum())
print("Count of fraud(1) and non-fraud(0) in  the given dataset:")
print(df['Class'].value_counts())

# sns.countplot(x='Class', data=df)
# plt.show()

# sns.pairplot(df_down_sampled)
# plt.show()

std_scaler = StandardScaler()
rob_scaler = RobustScaler()

df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1, 1))

df.drop(['Time', 'Amount'], axis=1, inplace=True)

scaled_amount = df['scaled_amount']
scaled_time = df['scaled_time']

df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
df.insert(0, 'scaled_amount', scaled_amount)
df.insert(1, 'scaled_time', scaled_time)

# df = df.sample(frac=1)

# amount of fraud classes 492 rows.
fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0][:492]

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

# Shuffle dataframe rows
new_df = normal_distributed_df.sample(frac=1, random_state=42)

new_df.head()
print(new_df['Class'].value_counts())

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

sns.countplot('Class', data=df_down_sampled)
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()

#Now, in the downsampled data, we try to understnd the correlation between deatures.

f, (ax1) = plt.subplots(1, figsize=(28, 24))
# corr = df.corr()
# sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size': 10}, ax=ax1)
# ax1.set_title("Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=5)
# plt.show()

sub_sample_corr = new_df.corr()
sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size': 10}, ax=ax1)
ax1.set_title('SubSample Correlation Matrix \n (use for reference)', fontsize=5)
plt.show()

















import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


### Reduce memory size
# The goal of the project is to predict the probabilities of transaction being fraud.
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem,
                                                                              100 * (start_mem - end_mem) / start_mem))
    return df


### Load data
train_identity = pd.read_csv('train_identity.csv')
train_transaction = pd.read_csv('train_transaction.csv')
test_identity = pd.read_csv('test_identity.csv')
test_transaction = pd.read_csv('test_transaction.csv')

print(train_identity.head())
print(train_transaction.head())

# As we can see, that transactionId exists in both datasets. So let's combine the data and identity and transactions
# dataset by joining this TransactionId column
train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')

# Now lets check the total number of rows and columns of the finalized dataset.
print(train.shape)
print(test.shape)
# print(train.head())
# print(train.describe())

# Lets Check the class values distribution
sns.countplot(x='isFraud', data=train)
plt.show()
train_fraud = train['isFraud']
print(train['isFraud'].value_counts(normalize=True))

# reduce_mem_usage(train)
# reduce_mem_usage(test)
# As we have merged the identity and trasactions data, lets delete the other dataframes to save memory.
del train_identity, train_transaction, test_identity, test_transaction

### Data cleaning and selection

print(train.isnull().sum())
print('null values in test data:')
print(test.isnull().sum())

# Work on Categorical values
categorical_columns = train.select_dtypes(include=['object']).columns
print(categorical_columns)
# Now we are creating a new dataframe to identify the columns with no useful data or no values at all.
# We can delete all the columns with only null/nan values
data_null = train.isnull().sum() / len(train) * 100
data_null.drop(data_null[data_null == 0].index).sort_values(ascending=False)[:500]
missing_data = pd.DataFrame({'Missing Ratio': data_null})
missing_data.head()


def get_too_many_null_attr(data):
    many_null_cols = [col for col in data.columns if data[col].isnull().sum() / data.shape[0] > 0.9]
    return many_null_cols


def get_too_many_repeated_val(data):
    big_top_value_cols = [col for col in train.columns if
                          data[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
    return big_top_value_cols


print(train.shape)
too_many_null = get_too_many_null_attr(train)
print("More than 90% null: " + str(len(too_many_null)))
train.drop(too_many_null, axis=1, inplace=True)
too_many_repeated = get_too_many_repeated_val(train)
print("More than 90% repeated value: " + str(len(too_many_repeated)))
train.drop(too_many_repeated, axis=1, inplace=True)
print(train.shape)
print(list(train.columns))

# Add back the isFraud class label
train['isFraud'] = train_fraud

# As seen above, we are able to reduce the number of dimensions from 434 to 355 just by eliminating the null columns and highly repaeated
# same values in the columns.

### Exploratory Data Analysis
# 1. Lets check the times, days and months  on which the frauds happened more often
# The Transaction dates in the dataset are just the relative times from the starting transaction.
# 86400 // 24 * 60 * 60
start_date = datetime.datetime.strptime('2019-10-30', "%Y-%m-%d")
train['Date'] = train['TransactionDT'].apply(lambda x: (start_date + datetime.timedelta(seconds=x)))
train['_year_month'] = train['Date'].dt.year.astype(str) + '-' + train['Date'].dt.month.astype(str)
train['_weekday'] = train['Date'].dt.dayofweek
train['_hour'] = train['Date'].dt.hour
train['_day'] = train['Date'].dt.day

fig, ax = plt.subplots(4, 1, figsize=(16, 12))

train.groupby('_weekday')['isFraud'].mean().to_frame().plot.bar(ax=ax[0])
train.groupby('_hour')['isFraud'].mean().to_frame().plot.bar(ax=ax[1])
train.groupby('_day')['isFraud'].mean().to_frame().plot.bar(ax=ax[2])
train.groupby('_year_month')['isFraud'].mean().to_frame().plot.bar(ax=ax[3])

plt.show()

# 2. Lets check the amounts ranges on which there are more frauds
# Discretize the amounts into 15 buckets and visualize on which buckets, there are more frauds
train['amount_buckets'] = pd.qcut(train['TransactionAmt'], 15)
df = train.groupby('amount_buckets')['isFraud'].mean().to_frame()
df.sort_values(by='isFraud', ascending=False)
print(df)

fig, ax = plt.subplots(1, 1, figsize=(16, 12))
df.plot.bar()
plt.show()

# Device Type Analysis
train_fraud_subset = train[train.isFraud==True]
sns.countplot(x='DeviceType', data=train_fraud_subset)
plt.show()

# Device Info Analysis
device_info_plots = sns.countplot(x='DeviceInfo', data=train_fraud_subset)
device_info_plots.set_xticklabels(device_info_plots.get_xticklabels(), rotation=90)
plt.show()

# Browser Type Analysis
browser_count_plot = sns.countplot(x='id_31', data=train_fraud_subset)
browser_count_plot.set_xticklabels(browser_count_plot.get_xticklabels(), rotation=90)
plt.show()

# Browser Aggregations based on its product/company name
train_fraud_subset['browser_name'] = train_fraud_subset['id_31'].apply(lambda x: x.split()[0] if x == x else 'unknown')
browser_count_plot = sns.countplot(x='browser_name', data=train_fraud_subset)
browser_count_plot.set_xticklabels(browser_count_plot.get_xticklabels(), rotation=90)
plt.show()

# Todo: Don't attempt to delete ip type(id_23) from data cleaning
# browser_count_plot = sns.countplot(x='id_23', data=train_fraud_subset)
# browser_count_plot.set_xticklabels(browser_count_plot.get_xticklabels(), rotation=90)
# plt.show()
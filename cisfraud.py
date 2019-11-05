import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train_identity = pd.read_csv('train_identity.csv')
train_transaction = pd.read_csv('train_transaction.csv')
test_identity = pd.read_csv('test_identity.csv')
test_transaction = pd.read_csv('test_transaction.csv')
sub = pd.read_csv('sample_submission.csv')
# let's combine the data and work with the whole dataset
train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')

# print(train.head())
# print(train.describe())

# sns.countplot(x='isFraud', data=train)
# plt.show()

print(train.isnull().sum())
print('null values in test data:')
print(test.isnull().sum())

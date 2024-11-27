import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder


def extract_title(name):
    import re
    title_search = re.search(' ([A-Za-z]+)\\.', name)
    if title_search:
        return title_search.group(1)
    return ""


def fill_cabin(row):
    return 'U' if pd.isna(row['Cabin']) else row['Cabin'][0]


def adjust_pclass(row):
    return row['Pclass']


def preprocess(df):
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    df['Title'] = df['Name'].apply(extract_title)
    df['Title'] = df['Title'].replace(['Mlle', 'Ms', 'Mme'], 'Miss')
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                       'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                       'Jonkheer', 'Dona'], 'Rare')

    title_ages = df.groupby('Title')['Age'].median()
    for title, age in title_ages.items():
        df.loc[(df['Age'].isnull()) & (df['Title'] == title), 'Age'] = age

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    df['IsAlone'] = 1
    df['IsAlone'].loc[df['FamilySize'] > 1] = 0

    df['FarePerPerson'] = df['Fare'] / df['FamilySize']

    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])

    df = pd.get_dummies(df, columns=['Embarked', 'Pclass'], drop_first=True)

    title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'] = df['Title'].fillna(0)
    #df = df.drop(['Ticket', 'Name', 'Cabin'],axis=1)
    df = df.drop(['Ticket', 'Name', 'Cabin','Age','SibSp','Parch','FamilySize'], axis=1)
    return df

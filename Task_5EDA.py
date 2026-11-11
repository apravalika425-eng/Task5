import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style='whitegrid')
os.makedirs('eda_outputs', exist_ok=True)

df = pd.read_csv('titanic_dataset.csv')
print(df.shape)
print(df.info())
print(df.describe(include='all').to_string())
print(df.isnull().sum())
print(df['Survived'].value_counts())
print(df['Sex'].value_counts())
print(df['Pclass'].value_counts())
print(df['Embarked'].value_counts())

plt.figure()
sns.histplot(df['Age'].dropna(), kde=True, bins=30)
plt.title('Histogram: Age Distribution')
plt.savefig('eda_outputs/01_histogram_age.png')
plt.show()

plt.figure()
sns.histplot(df['Fare'], kde=True, bins=30)
plt.title('Histogram: Fare Distribution')
plt.savefig('eda_outputs/02_histogram_fare.png')
plt.show()

plt.figure()
sns.boxplot(x='Pclass', y='Fare', data=df)
plt.title('Boxplot: Fare vs Pclass')
plt.savefig('eda_outputs/03_boxplot_fare_pclass.png')
plt.show()

plt.figure()
sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Boxplot: Age vs Survival')
plt.savefig('eda_outputs/04_boxplot_age_survived.png')
plt.show()

plt.figure()
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Survival by Sex')
plt.savefig('eda_outputs/05_countplot_sex_survived.png')
plt.show()

plt.figure()
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival by Class')
plt.savefig('eda_outputs/06_countplot_pclass_survived.png')
plt.show()

plt.figure()
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df, alpha=0.6)
plt.title('Scatter: Age vs Fare')
plt.savefig('eda_outputs/07_scatter_age_fare.png')
plt.show()

plt.figure(figsize=(8,5))
sns.heatmap(df[['Survived','Pclass','Age','SibSp','Parch','Fare']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap: Correlation')
plt.tight_layout()
plt.savefig('eda_outputs/08_heatmap_correlation.png')
plt.show()

pair = sns.pairplot(df[['Age','Fare','Pclass','Survived']], hue='Survived')
pair.savefig('eda_outputs/09_pairplot.png')
plt.show()

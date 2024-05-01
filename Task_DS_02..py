
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
titanic_data = pd.read_csv("titanic\gender_submission.csv")

# Check for missing values
print(titanic_data.isnull().sum())

# Perform data cleaning steps such as handling missing values and removing irrelevant rows
titanic_data.dropna(subset=['PassengerId',"Survived"], axis=0, inplace=True)

# Explore the statistical summary of the dataset
print(titanic_data.describe())

# Visualize the data to identify patterns and trends

titanic_data['PassengerId'].hist()
plt.xlim(890, 1310)
plt.ylim(0)
# Hide y-axis scale
plt.yticks([])
plt.xlabel('PassengerId')
plt.title('Passenger')
plt.show()

titanic_data['Survived'].hist()
plt.ylim(0)
# Hide y-axis scale
plt.xticks([0, 1])
plt.yticks([])
plt.xlabel('Survived')
plt.title('Survived')
plt.show()


# Analyze the relationship between variables
sns.heatmap(titanic_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


#######################################################################################

titanic_data_Test = pd.read_csv("titanic/train.csv")

print(titanic_data_Test.describe())

# Visualize the data to identify patterns and trends

titanic_data_Test['Age'].hist()
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Age')
plt.show()

# Explore the distribution of categorical variables
sns.countplot(x='Survived', hue='Sex', data=titanic_data_Test
)
plt.title('Survival Count by Gender')
plt.show()

#################################################################################

titanic_data_Test1 = pd.read_csv("titanic/train.csv")


# Identify trends in the data

sns.lineplot(x='Age', y='Fare', data=titanic_data_Test1)
plt.title('Trend between Age and Fare')
plt.show()

# Explore survival rates based on different variables

sns.barplot(x='Pclass', y='Survived', data=titanic_data_Test1)
plt.title('Survival Rate by Passenger Class')
plt.show()

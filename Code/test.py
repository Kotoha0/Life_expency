# Step 1: Read the data from the CSV file
import pandas as pd

# Read the CSV file into a DataFrame
data = pd.read_csv('LifeExpectance.csv')

# Step 2: Split the data into train and test datasets
from sklearn.model_selection import train_test_split

# Splitting data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Step 3: Basic information about the data
# a. Number of records in each dataset
print("Number of records in train dataset:", len(train_data))
print("Number of records in test dataset:", len(test_data))

# b. Histogram and statistical information about life expectancy
import matplotlib.pyplot as plt

plt.hist(train_data['LifeExpectancy'], bins=20, color='skyblue')
plt.title('Histogram of Life Expectancy')
plt.xlabel('Life Expectancy')
plt.ylabel('Frequency')
plt.show()

print("Statistical information about life expectancy:")
print(train_data['LifeExpectancy'].describe())

# c. Find three countries with the highest life expectancy
top_countries = train_data.nlargest(3, 'LifeExpectancy')[['Country', 'LifeExpectancy']]
print("Three countries with the highest life expectancy:")
print(top_countries)

# Step 4: Fit linear regression models
from sklearn.linear_model import LinearRegression

# Fit models for GDP, Total expenditure, and Alcohol
features = ['GDP', 'TotalExpenditure', 'Alcohol']
models = {}
for feature in features:
    model = LinearRegression()
    model.fit(train_data[[feature]], train_data['LifeExpectancy'])
    models[feature] = model

# Step 5: Find coefficients and scores of regression lines
for feature, model in models.items():
    print(f"\nModel for {feature}:")
    print("Coefficient (slope):", model.coef_[0])
    print("Intercept:", model.intercept_)
    print("Score:", model.score(train_data[[feature]], train_data['LifeExpectancy']))

    # Plotting data points and regression line
    plt.scatter(train_data[feature], train_data['LifeExpectancy'], color='blue', label='Data points')
    plt.plot(train_data[feature], model.predict(train_data[[feature]]), color='red', label='Regression line')
    plt.title(f'Regression line for {feature}')
    plt.xlabel(feature)
    plt.ylabel('Life Expectancy')
    plt.legend()
    plt.show()

# Step 6: Predict life expectancy for the test set
from sklearn.metrics import mean_squared_error

errors = []
for feature, model in models.items():
    predictions = model.predict(test_data[[feature]])
    error = mean_squared_error(test_data['LifeExpectancy'], predictions)
    errors.append(error)

# Calculate average error and standard deviation
average_error = sum(errors) / len(errors)
std_deviation = (sum((error - average_error) ** 2 for error in errors) / len(errors)) ** 0.5

print("\nAverage error for all three models:", average_error)
print("Standard deviation for predictions:", std_deviation)

# Step 7: Prepare report
print("\nReport:")
print("Number of records in train dataset:", len(train_data))
print("Number of records in test dataset:", len(test_data))
print("\nStatistical information about life expectancy:")
print(train_data['LifeExpectancy'].describe())
print("\nThree countries with the highest life expectancy:")
print(top_countries)
print("\nAverage error for all three models:", average_error)
print("Standard deviation for predictions:", std_deviation)

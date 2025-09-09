import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
# --- 1. Synthetic Data Generation ---
np.random.seed(42)  # for reproducibility
# Generate synthetic customer data
num_customers = 500
data = {
    'customer_id': range(1, num_customers + 1),
    'tenure_months': np.random.randint(1, 25, num_customers),
    'avg_monthly_spend': np.random.uniform(10, 100, num_customers),
    'support_tickets': np.random.poisson(2, num_customers),
    'social_media_sentiment': np.random.choice([-1, 0, 1], num_customers, p=[0.2, 0.6, 0.2]), # -1: negative, 0: neutral, 1: positive
    'churned': np.random.binomial(1, 0.2, num_customers) # 20% churn rate
}
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Feature Engineering ---
# No significant cleaning needed for synthetic data.  Feature engineering could be more extensive with real data.
df['overall_sentiment'] = df['social_media_sentiment'] * 0.5 + df['support_tickets'] * -0.2 #Example weighting
# --- 3. Data Analysis and Model Training ---
# Separate features (X) and target (y)
X = df[['tenure_months', 'avg_monthly_spend', 'overall_sentiment']]
y = df['churned']
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a logistic regression model (simple model for demonstration)
model = LogisticRegression()
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
# --- 4. Visualization ---
# Create a scatter plot of average monthly spend vs. tenure, colored by churn status
plt.figure(figsize=(8, 6))
sns.scatterplot(x='tenure_months', y='avg_monthly_spend', hue='churned', data=df, palette=['blue', 'red'])
plt.title('Customer Churn by Tenure and Spending')
plt.xlabel('Tenure (Months)')
plt.ylabel('Average Monthly Spend')
plt.savefig('churn_scatter.png')
print("Plot saved to churn_scatter.png")
# Create a bar plot of churn rate by overall sentiment
plt.figure(figsize=(8, 6))
sns.barplot(x='overall_sentiment', y='churned', data=df, estimator=lambda x: sum(x)*100/len(x), ci=None)
plt.title('Churn Rate by Overall Sentiment')
plt.xlabel('Overall Sentiment')
plt.ylabel('Churn Rate (%)')
plt.savefig('churn_sentiment.png')
print("Plot saved to churn_sentiment.png")
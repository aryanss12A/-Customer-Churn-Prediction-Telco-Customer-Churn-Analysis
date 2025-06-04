import pandas as pd
import mysql.connector
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Connect to MySQL and load data
conn = mysql.connector.connect(
    host="localhost", user="root", password="root123", database="churn_analysis"
)
query = "SELECT * FROM customers"
df = pd.read_sql(query, conn)
conn.close()

# Clean TotalCharges and drop missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Drop customerID
df.drop("customerID", axis=1, inplace=True)

# Replace 'No internet service' and 'No phone service' with 'No'
cols_no_internet = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for col in cols_no_internet:
    df[col] = df[col].replace({'No internet service': 'No'})
df['MultipleLines'] = df['MultipleLines'].replace({'No phone service': 'No'})

# Encode target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Visualization 1: Churn Distribution Pie Chart
plt.figure(figsize=(5, 5))
plt.pie(df['Churn'].value_counts(), labels=['No Churn', 'Churn'], autopct='%1.1f%%',colors=['#66b3ff','#ff9999'])
plt.title('Churn Distribution')
plt.show()

# One-hot encode categorical features
X = pd.get_dummies(df.drop('Churn', axis=1))
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Random Forest with hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [8, 12, 16],
    'min_samples_split': [2, 4]
}
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
grid = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

# Predict
y_pred = grid.predict(X_test)

# Visualization 2: Feature Importance
importances = grid.best_estimator_.feature_importances_
indices = importances.argsort()[-10:][::-1]
plt.figure(figsize=(8, 5))
sns.barplot(x=importances[indices], y=[X.columns[i] for i in indices], palette='viridis')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importance')
plt.show()
 
# Visualization 3: Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Visualization 4: ROC Curve
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, grid.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2,linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
# Accuracy
print("Best params:", grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

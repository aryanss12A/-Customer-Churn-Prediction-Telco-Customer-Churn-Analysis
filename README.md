
# -Customer-Churn-Prediction-Telco-Customer-Churn-Analysis-


Churn refers to the rate at which customers or subscribers discontinue their relationship with a company or service. It's a critical metric for businesses as it directly impacts revenue, growth, and overall customer retention.

This project predicts customer churn for a telecom company using the Telco Customer Churn dataset. The goal is to identify customers likely to leave the company, enabling targeted retention strategies.

🚀 Key Features

End-to-End Machine Learning Pipeline:
Data cleaning, feature engineering, model training, evaluation, and visualization.

Data Preprocessing:
1.Handled missing values and outliers
2.Encoded categorical features using one-hot encoding
3.Scaled numerical features

Modeling:
1.Built and tuned a Random Forest Classifier using GridSearchCV
2.Achieved ~79% accuracy and 0.83 ROC-AUC score

Visualization:
1.Churn distribution pie chart
2.Confusion matrix heatmap
3.ROC curve with AUC
4.Feature importance bar plot

🛠️ Technologies Used :
1.Python 3.12
2.pandas, numpy
3.scikit-learn
4.matplotlib, seaborn
5.MySQL (for data storage/retrieval)

Results :
1.Best Model: Random Forest Classifier
2.Test Accuracy: ~79%
3.ROC-AUC: 0.83

📊 Visualizations
1. Churn Distribution Pie Chart
This chart shows the percentage of customers who churned versus those who did not.
It helps visualize class imbalance in the dataset.
![Churn Distribution Pie Chart](https://github.com/user-attachments/assets/12d80dc9-3810-41e1-a033-1cb 

2. Confusion Matrix
The confusion matrix heatmap displays the number of correct and incorrect predictions for both churned and non-churned customers.
It helps assess the model’s classification performance.
![Confusion Matrix](https://github.com/user-attachments/assets/4b693e6d-1593-4e0f-9130-904e 

3. ROC Curve
The ROC (Receiver Operating Characteristic) curve illustrates the trade-off between the true positive rate and false positive rate at various thresholds.
The area under the curve (AUC) quantifies the model's ability to distinguish between classes.
![ROC Curve](https://github.com/user-attachments/assets/fb5fa086-3ce3-445f-9d1b-3b2070b

4. Feature Importance
This bar plot ranks the top features that contribute most to the churn prediction, as determined by the Random Forest model.
It provides insight into which customer attributes are most influential.
![Feature Importance](https://github.com/user-attachments/assets/fc09b493-320b-4516-b03c-0f16169d 

💡 Business Insights
1.Identified key drivers of churn such as contract type, tenure, and monthly charges.
2.Provided actionable recommendations for customer retention based on model insights.

📚 References
Telco Customer Churn Dataset (Kaggle)

👤 Author
Aryan Sachdeva






# Project Report
## Customer Churn Prediction and Retention Strategies
*Using Machine Learning Models and Evaluation of their Performance*


### Application: 
- Programming: Python (Pandas, Numpy, Scikit-Learn)
- Data Visualization: Python (Matplotlib, Seaborn)

### Introduction:
Customer churn threatens subscription businesses, impacting revenue and growth. This project analyzes customer behavior to identify churn drivers and build predictive models for churn likelihood, enabling targeted retention strategies that enhance customer lifetime value. By identifying at-risk customers, businesses can ensure revenue stability, improve marketing ROI, and refine product offerings based on root-cause analysis.

### Data Description: 
Sourced from Kaggle, the dataset includes 440,833 rows and 12 columns (3 categorical, 8 numerical), with a binary churn target. Key features cover:
- Customer Demographics: Age, Gender
- Engagement Metrics: Tenure, Usage Frequency, Support Calls
- Payment Behavior: Payment Delay, Total Spend
- Subscription Details: Subscription Type, Contract Length
- Customer Interaction: Last Interaction
- Target Variable: Churn (Binary indicator of whether a customer left)

### Data Cleaning and Exploratory Data Analysis (EDA):
After addressing missing values and ensuring all customers were unique, we used SMOTE to balance the mild class imbalance. Correlation analysis revealed that support_calls and total_spend are linked to churn, with total_spend negatively correlated—higher spending is associated with lower churn. Variables like payment delays, age, and last interaction showed moderate correlation with churn.
Churn rates were highest among standard, basic, and premium subscription types, with monthly contracts having the highest churn, while quarterly and annual contracts showed lower churn. Customers who churned typically spent less on average.
Churn probability sharply increases after a payment delay of 20 days, with delays under 20 days having lower churn rates. Age group analysis revealed that young professionals are equally likely to churn or stay, young adults and middle-aged customers tend to churn more, established adults stay more, and seniors all churn.

### ML Models:
The training data was preprocessed by applying One Hot Encoding to categorical columns (gender, subscription type, contract length). Numerical columns were left unchanged for tree-based models, while Standard Scaling was applied for Logistic Regression and MLP Classifier models. Customer churn was predicted and evaluated using the following machine learning models:

## Logistic Regression
We initially used a basic machine learning technique, Logistic Regression, to build the training data. Various hyperparameters were tuned using 5-fold cross-validation. The best model obtained from the training set was then evaluated on the test set to assess how well it generalizes.

## Hyperaparameter tuning:
- Penalty: Regulates regularization to prevent overfitting. Tested penalties: Lasso (l1), Ridge (l2), Elastic Net
- Regularization strength (C): Smaller values of C help decrease model complexity. Tested values: 0.0001, 0.001, 0.005, 0.01, 0.1, 1
- Solver: Optimization algorithms to minimize the loss function. Tested options: lbfgs, liblinear, saga
- l1_ratio (for Elastic Net): Balances contributions of Lasso and Ridge penalties. Tested values: 0.1, 0.5, and 0.9 

### Model Performance
The performance of the Logistic Regression model was evaluated on both the train and test sets. The results are summarized in the table below:

*Note: Out of all the combinations we have tried, the performance remains the same overall.*

![LR Image](Screenshot 2025-04-02 at 11.30.52 PM)

### Training Set Performance
- The model achieved 90% accuracy on the training set, which indicates that it fits the training data well
- A precision of 93% shows that when the model predicts a positive class, it is correct 93% of the time
- A recall of 87% suggests that the model captures most of the actual positive cases
- The F1-score of 0.90 reflects a good balance between precision and recall, indicating that the model performs well on the training set

### Test Set Performance
- The test set accuracy drops significantly to 58%, indicating that the model struggles to generalize to unseen data
- The precision of 53% on the test set suggests that nearly half of the positive predictions are incorrect, leading to a high false positive rate
- The recall is extremely high at 99%, indicating that the model identifies almost all positive cases but at the cost of misclassifying many negative cases as positive
- The F1-score of 0.69 reflects a moderate balance between precision and recall, but the low precision points to issues with overfitting

### Conclusion
The 32% drop in accuracy from training to test indicates overfitting. The high recall and low precision on the test set show the model overpredicts positives, resulting in many false positives and poor generalization.

## Decision Tree
To build a robust classification model, we employed a Decision Tree Classifier while carefully tuning its hyperparameters to reduce overfitting and improve generalization. We experimented with various values for key hyperparameters using GridSearchCV with 5-fold cross-validation, iterating over different configurations to optimize model performance.

## Hyperparameter Tuning:
We tuned the following parameters to balance complexity and generalization:
- Max Depth: Controlled the depth of the tree to prevent overfitting. Limiting depth to 3 helped reduce model complexity.
- Min Samples Split: Ensured that each split occurred only if there were at least 30 samples, preventing small, unreliable splits.
- Min Samples Leaf: Restricted leaf nodes to contain at least 40 samples, further reducing variance.
- CCP Alpha: Applied pruning (ccp_alpha = 0.08) to eliminate unnecessary branches, enhancing model generalization.
- The best hyperparameter combination was: {‘ccp_alpha’: 0.08, ‘max_depth’: 3, ‘min_samples_leaf’: 40, ‘min_samples_split’: 30}

### Model Performance:

![DT Image](Screenshot 2025-04-02 at 11.31.12 PM)

### Training Set Performance:
- Achieved 0.87 accuracy, indicating a good fit to training data.
- Precision of 0.99 shows that most predicted positives were correct.
- Recall 0.78 suggests that most actual positives were detected.
- F1-score of 0.87 indicates a strong balance between precision and recall.

### Test Set Performance:
- Accuracy dropped to 0.61, highlighting overfitting to the training data.
- Precision for Class 1 is 0.55, meaning that nearly half of predicted positives are false positives.
- Recall is 0.93, meaning the model captures most positive cases but misclassifies many negative cases.
- F1-score of 0.57 reflects an imbalance between precision and recall.
- AUC score of 0.567 shows weak separability between classes.

### Conclusion
The Decision Tree model performs well on the training set but struggles to generalize to the test data. The high recall and low precision suggest that the model overpredicts positive cases, leading to a high false positive rate. Despite hyperparameter tuning and pruning, the overfitting issue persists, as evidenced by the drop in accuracy and F1-score on the test set.

## Gradient Boosting
To improve our model's performance, we employed Gradient Boosting, an ensemble machine learning technique that builds models sequentially to correct previous errors. We tuned multiple hyperparameters using GridSearchCV with 5-fold cross-validation to find the best combination.

### Hyperparameter Tuning:
- Number of estimators (n_estimators): Defines the number of boosting stages. We experimented with values ranging from 10 to 500.
- Learning rate (learning_rate): Controls the contribution of each tree to the final model. We tested values from 0.002 to 0.1.
- Max depth (max_depth): Limits the depth of individual trees to control model complexity, with values from 1 to 5.
- Feature selection (max_features): We tried "sqrt" and "log2" for feature selection per split.
- Subsample ratio (subsample): Introduces randomness by using a fraction of training data for each tree, with values from 0.6 to 0.8.
- Minimum samples split (min_samples_split): Defines the minimum number of samples required to split a node. Values of 50 and 100 were tested.
- After tuning, the best hyperparameters found were:{'learning_rate': 0.002, 'max_depth': 1, 'min_samples_split': 50, 'n_estimators': 10, 'subsample': 0.6}


### Model Performance:

![GB Image](Screenshot 2025-04-02 at 11.31.22 PM)

### Training Set Performance
- The model achieved 75% accuracy, indicating that it learns patterns in the training data effectively.
- A precision of 84% shows that when predicting the positive class, the model is correct most of the time.
- The recall of 75% suggests that the model captures a good proportion of actual positive cases.
- The F1-score of 0.75 balances precision and recall.

### Test Set Performance
- The test accuracy of 66% is lower than the training accuracy, but the gap is not extreme, suggesting a reasonable generalization capability.
- The precision (68%) and recall (67%) indicate that the model can distinguish positive and negative classes better than random guessing but still misclassifies some cases.
- The F1-score of 0.66 shows a balance between precision and recall.
- The ROC-AUC score of 0.6678 suggests moderate discriminatory power.

### Conclusion
Gradient Boosting provided better generalization than earlier models but still struggles with imbalanced class predictions. While the ROC-AUC score (0.6678) indicates a moderate ability to distinguish between classes, tuning hyperparameters further—such as adjusting subsampling, learning rate decay, or adding class weighting—might improve performance.

## MLP
To explore the potential of neural networks for our classification problem, we implemented a Multi-Layer Perceptron (MLP) model. MLP is a type of feedforward artificial neural network that learns non-linear relationships through multiple hidden layers. We fine-tuned several hyperparameters using GridSearchCV with 5-fold cross-validation to optimize performance.

### Hyperparameter Tuning:
- Hidden layer sizes (hidden_layer_sizes): Defines the structure of the neural network. We tested architectures like (128, 64), (256, 128, 64), (32,) and others to balance model complexity.
- Activation function (activation): We used ReLU (Rectified Linear Unit), which helps introduce non-linearity into the model.
- Solver (solver): Determines how weights are optimized. We primarily tested "adam" and "lbfgs".
- Regularization (alpha): Prevents overfitting by adding an L2 penalty. Values tested ranged from 0.0001 to 1000.0.
- Maximum iterations (max_iter): Defines the number of training epochs. We experimented with values between 50 and 500.
- After tuning, the best hyperparameters were:{'activation': 'relu', 'alpha': 100.0, 'hidden_layer_sizes':(64,), 'max_iter': 500, 'solver': 'adam'}

### Model Performance:

![MLP Image](Screenshot 2025-04-02 at 11.31.33 PM)

### Training Set Performance
- The training accuracy of 89% indicates strong learning capability.
- Precision (92%) and recall (86%) suggest somewhat balanced classification of positive and negative cases.
- The F1-score of 0.89 shows a strong trade-off between precision and recall.

### Test Set Performance (Using Adjusted Threshold)
- The test accuracy of 68% suggests reasonable generalization to unseen data.
- Precision (66%) and recall (66%) are well-balanced, indicating moderate classification performance.
- The F1-score of 0.66 suggests an acceptable trade-off.
- The ROC-AUC score of 0.75 indicates better discrimination between classes compared to previous models.

### Conclusion
- The MLP model demonstrated strong learning ability but exhibited signs of overfitting to the training data, as indicated by the high training accuracy (89%) and a lower test accuracy (68%). Adjusting the decision threshold improved test accuracy slightly. The ROC-AUC of 0.75 indicates a better trade-off between sensitivity and specificity compared to previous models.

### Final Model Selection
- If the business prioritizes accuracy and generalization, we recommend Gradient Boosting as it provides a balanced trade-off between precision and recall while avoiding significant overfitting.
- If the business prioritizes recall (identifying as many churn customers as possible), we recommend MLP Classifier, which has a higher ROC-AUC score (0.75) and better generalization, making it more suitable for capturing complex patterns in customer behavior.

### Insights
Frequent support calls indicate dissatisfaction, making proactive customer support crucial. Late payments significantly increase churn risk, highlighting the need for payment reminders via SMS, email, and push notifications. Lower-spending customers and those on monthly contracts are more likely to churn, suggesting that loyalty rewards and contract incentives could improve retention. Younger customers are more price-sensitive, while older customers expect higher service quality, emphasizing the need for customized plans. Shorter-tenure customers face higher churn, stressing the importance of early engagement.

### Recommendations
To address these issues, businesses should implement AI-powered chat support to enhance service efficiency, resolve issues faster, and reduce frustration. Offering exclusive benefits for monthly contract renewals and personalizing retention strategies based on customer data—such as spending habits and interactions—can help retain high-value customers. Onboarding incentives and personalized check-ins can improve early engagement, ensuring long-term customer loyalty. These data-driven interventions will enhance customer satisfaction and reduce churn effectively.








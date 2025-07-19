# Senior AI/ML Engineer Assessment: Insurance Claims Fraud Detection
## 🔍 Problem Context
InsuranceCorp's fraud detection system flags too many legitimate claims as fraudulent (35% false positive rate), causing customer service issues and payment delays. Your job: build an optimized model to reduce false positives while maintaining fraud detection capability.

## 📂 File Structure
fraud-detection/  
├── data/  
│   └── claims.csv (PROVIDED)  
├── fraud_model.py (EDIT THIS)  
└── QUESTIONS.md (EDIT THIS)  

## 🧪 How to Test
This is some text.

```
# Install dependencies
pip install pandas numpy scikit-learn matplotlib

# Run your model
python fraud_model.py
```

## [REQUIRED] ✅ Task: Build Optimized Fraud Detection Model

__Edit:__ <span style="color: lightblue"> __fraud_model.py__ </span>

Build and optimize a fraud detection model:
1.	__Load and explore__ the claims dataset
2.	__Preprocess the data__ (handle missing values, outliers, feature scaling)
3.	__Train multiple models__ (at least 2 different algorithms like `sklearn.linear_model.LogisticRegression, sklearn.ensemble.RandomForestClassifier, sklearn.ensemble.GradientBoostingClassifier`, etc.)
4.	__Evaluate and compare__ using precision, recall, and F1-score
5.	__Select the best model__ and explain your choice

__Success Criteria:__
* Achieve precision > 0.75 (reduce false positives)
* Maintain recall > 0.60 (still catch fraud)
* Clear model comparison with business impact explanation

__Expected Output:__ Your script should print a summary like:
```
Model Performance Comparison:
Random Forest: Precision=0.78, Recall=0.65, F1=0.71
Logistic Regression: Precision=0.82, Recall=0.58, F1=0.68

Recommended Model: Random Forest
Business Impact: Reduces false positives by 40% while maintaining fraud detection
```

## [REQUIRED] 📄 Written Questions
__Edit:__ <span style="color: lightblue"> __QUESTIONS.md__ </span>

__Question 1: LLM Integration Strategy__
Design how you would integrate Large Language Models into this fraud detection system:
* How would LLMs analyze claim descriptions for fraud indicators?
* What prompt engineering approach would you use for insurance claims?
* Which LLM would you choose (GPT-4, Claude, etc.) and why?

__Question 2: Production Deployment Architecture__
Describe your approach to deploying this model in production on any cloud provider (e.g. AWS, Azure, GCP):
* Which services or components would you use to deploy and scale the model, and why?
* How would you handle real-time scoring for 100K+ daily insurance claims?
* What strategies would you use to monitor the system and detect model drift over time?

## 📜 Sample Data
__claims.csv contains:__  
```
claim_id,claim_amount,policy_age_days,claimant_age,claim_type,description,previous_claims,is_fraud
CLM001,2500.50,365,34,auto,"Minor fender bender",0,0
CLM002,15000.00,30,45,auto,"Total loss accident",2,1
```

__Key Features:__  
* <span style="color: brown"> __claim_amount__ </span>: Dollar amount of claim
* <span style="color: brown"> __policy_age_days__ </span>: How long customer has had policy
* <span style="color: brown"> __claimant_age__ </span>: Age of person filing claim
* <span style="color: brown"> __previous_claims__ </span>: Number of prior claims
* <span style="color: brown"> __is_fraud__ </span>: Target variable (1=fraud, 0=legitimate)
* <span style="color: brown"> __claim_type__ </span> is a categorical feature with a small number of values (e.g. 'auto', 'home', 'health'). Feel free to encode it using label encoding or one-hot encoding.

__Business Context:__  
* High claim amounts (>$10K) often require extra scrutiny
* Customers with multiple previous claims have higher fraud risk
* New policies (low policy_age_days) combined with high claims are suspicious
* Balance is critical: false positives anger customers, missed fraud costs money

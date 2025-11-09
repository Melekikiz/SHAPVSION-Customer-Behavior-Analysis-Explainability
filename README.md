# SHAPVSION-Customer-Behavior-Analysis-Explainability

📌 Project Overview

This project demonstrates advanced customer investment analysis using machine learning with explainability. 
The main goal is to identify High Potential clients (top 15% investment potential) and provide actionable business insights.

Key Features:

Random Forest & Logistic Regression modeling

SHAP (SHapley Additive exPlanations) for model explainability

Customer segmentation (K-Means, k=6)

Feature engineering: AvgInvestmentSize, InteractionScore, RiskScore


📊 Dataset

Records: 1,000 customers (simulated demo)

Features: Age, AnnualIncome, InvestmentAmountLast12M, AvgInvestmentSize, InteractionScore, RiskScore, etc.

Preprocessing:

Negative investments set to 0

Income median imputed

Outliers capped at 99th percentile

Engineered Features: 16 features including AvgInvestmentSize and InvestmentActivityScore


⚙️ Modeling

Train/Test Split: 80/20

Target: Top 15% investment clients (HighPotential)

Models:

Logistic Regression → Accuracy 0.885, AUC 0.873

Random Forest → Accuracy 0.91, AUC 0.873 (better recall & precision for positive class)

Evaluation: ROC curves, feature importance ranking


🔍 SHAP Explainability

Global Feature Impact (Bar Plot):

AvgInvestmentSize & AnnualIncome → most influential

InteractionScore → moderate impact

RiskScore → minor effect

Beeswarm Plot:

Red = high values (positive contribution), Blue = low values (negative)

Highlights direction and distribution of feature influence across customers

Force Plot Example:

Customer #10: Income & AvgInvestmentSize push prediction to High Potential, Age & RiskScore mild negative effect


💡 Key Business Insights

Target High Value Segments: Focus on clients with high income & investment size

Enhance Engagement: InteractionScore drives premium conversion; design targeted campaigns

Re evaluate RiskScore: Low impact; review scoring logic

CRM Integration: Use Random Forest probabilities as lead scores for prioritization

Recommended Actions:

Prioritize clients in high value clusters and RF probability >0.8 for immediate advisory outreach

Maintain a “High Potential” CRM list refreshed monthly

Personal invitations for high-income, low activity clients

Recalibrate RiskScore scoring methodology


📈 Visualizations

Included in /reports/plots/:

Feature Importance (RF) –
<img width="2400" height="1500" alt="feature_importance" src="https://github.com/user-attachments/assets/164b6d93-7ffc-445e-9b65-924aa4663c94" />


SHAP Summary Bar – shap_summary_bar.png
<img width="2400" height="1500" alt="shap_summary_bar" src="https://github.com/user-attachments/assets/18c901b2-01e6-42fc-a92a-61338784bd21" />


SHAP Beeswarm – shap_beeswarm.png
<img width="2293" height="1022" alt="shap_beeswarm" src="https://github.com/user-attachments/assets/9f12f37a-de43-4dcd-b6e7-d9488aa8bb82" />

ROC Curve Comparison – model_roc_curve.png
<img width="1920" height="1440" alt="model_roc_curve" src="https://github.com/user-attachments/assets/4af2d9cc-35fa-41a8-aaa1-f44733fc4537" />

PDF Report
[SHAPVSION.pdf](https://github.com/user-attachments/files/23439208/SHAPVSION.pdf)




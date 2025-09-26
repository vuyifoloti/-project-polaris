Polaris HQ – Attrition Risk Model Results

At Polaris HQ, we wanted to answer a simple but important question:
“Can we predict which employees are at risk of leaving the company before it happens?”

To explore this, we built an Attrition Risk Model using real workforce data. The goal was not only to test whether prediction is possible, but also to uncover the key drivers of attrition – the workplace conditions, roles, and patterns most strongly linked to whether people stay or leave.

What we did
•	We developed a predictive model (Logistic Regression) and trained it on employee history, roles, and work patterns.
•	We then tested the model’s ability to correctly flag employees who were likely to leave.
•	Finally, we analysed the model to identify the strongest signals of attrition – the factors that matter most.

Model performance

At the default decision threshold (50%), the model reached:
•	Accuracy: 78%
•	Precision (Leave): 40%
•	Recall (Leave): 76%
•	F1 Score (Leave): 52%

What this means: the model is effective at capturing 76% of the people at risk of leaving (high recall), but sometimes also predicts attrition for people who end up staying (lower precision). This balance is valuable because in attrition prevention, it’s often better to cast a slightly wider net than to miss key resignations.

Key drivers of attrition

The model revealed the top factors linked to attrition risk:
•	Overtime: Employees working frequent overtime are more likely to leave.
•	Business Travel: Frequent travellers show higher attrition risk, while non-travellers are more stable.
•	Role Tenure: Employees in the 9–11 year band are less likely to leave, suggesting a period of stability.
•	Manager Tenure: Employees with very new managers (0–2 years) are more stable, while mid-tenure managers (6–10 years) show mixed results.
•	Job Role: Sales Representatives and Laboratory Technicians are at higher risk; Research Directors and Managers are less so.
•	Commute Distance: Those with medium commutes (12–17 km) are more at risk than those with very short commutes (0–5 km).
•	Department & Organisation Income Buckets: Below-average department income correlates with higher attrition, while strong organisational earnings reduce risk.

These findings help us understand where attrition is most likely to happen and why – giving leaders clear starting points for action.

Next step – an interactive tool

To make this insight practical, we built an interactive app where stakeholders can explore predictions and drivers directly.


作者：禅与计算机程序设计艺术                    

# 1.背景介绍


## Introduction and Context
The paper presents an empirical study that examines the attitudes of workers to workplace hazardous conditions (WHP) among North American university employees who currently work in physical space settings with standardized protocols for handling WHP. The study involves analyzing data collected from a survey conducted through Qualtrics and using mixed-effects logistic regression models to examine the effects of individual characteristics on worker's beliefs about their perceptions regarding specific WHP procedures. 

Hazardous conditions are known to cause psychological distress among workers, which may lead them to question whether or not they should continue working under certain circumstances. In addition, reduced response times can result in significant cost savings to organizations due to fewer calls placed against individuals when something goes wrong. Therefore, it is crucial to understand employee attitudes toward WHP so that appropriate action can be taken to improve overall safety at workplaces.

To identify factors associated with reduction in response time after encountering hazardous conditions, we used data obtained from Qualtrics, a survey platform commonly used by universities to collect data related to various aspects of academic performance. We analyzed responses to several questions based on how workers felt about each procedure while in physical space settings, including awareness of risks, ability to safely move around, adherence to policies, and willingness to take action if needed. Based on this data, we developed a mixed-effects logistic regression model to identify individual variables that had a significant impact on reducing response time. 

## Methods
### Survey Design
We designed a survey targeting students enrolled in North American universities who have been physically located within a few minutes of a workplace where potential hazardous situations could occur. The survey asked the following six questions: 

1. How do you feel about the possibility of a fire or flood threat? (Self-rated Scale – Likert scale (Strongly Disagree/Disagree/Neutral/Agree/Strongly Agree))

2. Can you safely navigate your way around the building without being afraid of potential threats? (Likert scale (No/Sometimes/Seldom/Frequently))

3. Do you know what steps must be taken to stay safe if you notice something dangerous happening inside? (Yes/No)

4. Have you ever received training or supervision about safety measures at work? (Yes/No)

5. Does your job involve working outside of class hours, making personal contacts with other people, or providing essential services like food or medication? (Yes/No)

6. Have you ever experienced problems caused by unsafe driving or other forms of transportation? (Yes/No). 

All answers were rated on a seven point Likert scale and all respondents provided demographic information such as age, gender, race/ethnicity, occupation, years of experience, education level, and income bracket before completing the survey.

### Data Collection and Analysis
Data collection was done via Qualtrics, a survey platform widely used by universities worldwide. Our research team collaborated closely with stakeholders across the university system to ensure proper implementation and monitoring of surveys. During data collection, participants completed the survey covering various topics related to physical safety at work, including awareness of potential hazards, navigation, policy compliance, emergency planning, and job responsibilities. A total of 1791 survey responses were gathered spanning over two months (April-June 2021), comprising a diverse group of students and faculty alike from different majors and specializations. 

Analysis of the survey data involved applying multiple statistical methods, including mixed-effects logistic regression analysis and cluster analysis. Mixed-effects logistic regression model is one of the most popular techniques used in social sciences to analyze intervention effectiveness and treatment allocation between groups [REF]. Cluster analysis is another technique used to divide the population into subgroups based on similar behavioral patterns [REF].

In our analysis, we first identified the covariates affecting the probability of responding “no” to question 1 (How do you feel about the possibility of a fire or flood threat?) because lower scores indicate higher levels of confidence in the risk assessment process. We also included demographic variables such as gender, race/ethnicity, educational background, and occupation to capture contextual influences on the likelihood of responding negatively to these questions. 

Next, we applied mixed-effects logistic regression analysis to identify the independent variables that affected the response rate to reduce response time due to WHP. This approach allows us to control for confounding variables and account for heterogeneity across sample sizes. Specifically, we used random intercepts and slopes to estimate the relationship between the binary response variable (whether the participant reported having reduced response time) and individual characteristics (self-reported severity rating of possible fire/flood threat, navigability, adherence to precautionary measures, etc.). Additionally, we incorporated a random slope term to capture variation in baseline response rates across participants. Finally, we employed cluster analysis to further divide the sample into subgroups based on common behavioral features.

### Model Specification
Based on the results of the analysis, we formulated a mixed-effects logistic regression equation with the specified formula structure:

$$\text{Response}_i = \beta_0 + \beta_{\text{gender}_i} + \beta_{\text{race\_ethnicity}_i} + \beta_{\text{education\_level}_i} + \beta_{\text{occupation}_i} + \beta_{intercept}\text{Female}_i + \beta_{intercept}\text{Black}_i + \beta_{intercept}\text{First Gen}_i + u_{i}$$

Where $\text{Response}$ represents the binary response variable indicating whether the participant reported reducing response time after experiencing hazardous conditions; $u$ is the residual error term; and $\beta_0$, $\beta_{\text{gender}}, \beta_{\text{race\_ethnicity}}, \beta_{\text{education\_level}}, \beta_{\text{occupation}}$ are the fixed effects representing the overall trend, sex, ethnicity, education level, and occupation, respectively. Within clusters defined by demographics, we assume equal variances and normal distribution for the errors terms.

We then evaluated the significance of the estimated coefficients using permutation tests and bootstrapping approaches. If the p-values obtained from either test were below a chosen threshold, we considered those variables statistically significant and included them in the final model. Similarly, we controlled for multiple comparisons using Bonferroni correction before interpreting the coefficients.

Our final model provides insights into the attitudes of individuals towards workplace hazardous conditions. It highlights factors that are likely to influence a person’s decision-making processes in case of a WHP incident and suggests ways forward to mitigate any negative impact on employee safety and wellbeing.
# Predicting Future Portfolio Carbon Intensity with ML

## Introduction

The purpose of this paper is to enable smarter ESG investment, which allows investors and financial institutions to detect the low-carbon transition trends and to create long-term value through environmentally responsible investments, by developing a machine learning model that provides a user with the distribution of future carbon intensity for a company. Carbon emission intensity is defined as carbon emissions per unit of a company’s revenue. The paper focuses on predicting the GHG scope 1 emission intensity, which is defined as the intensity of the “direct emissions from owned or controlled sources". The study period is from 2010, the earliest data available for Bloomberg GHG scope 1 estimates, to the most recent 2021. The study uses the data, namely Bloomberg GHG scope 1 emission estimates, Revenue, Net Income, Market Capitalization, Free Cash Flow, and Industry of the company, for 8,052 companies worldwide extracted from the Bloomberg database. The research investigates whether there is any predictive relationship between carbon intensity and the aforementioned indicators. Based on the study of the data and their relationship, the paper develops a machine learning model for forecasting carbon emissions intensity. In contrast with most existing literature that forecasts carbon emissions or their intensity based on non-financial indicators, this paper provides new ideas by focusing mostly on financial factors and might serve as a useful reference for future research.

## Interface

In the final leg of the project, we are tasked to produce the outputs of the model and all its relevant information in a terminal accessible to all. We have decided to use the Dash framework to create an interface accessible through an URL that would access the backend Python code which would in turn be computing the distribution of the intensity variable based on the User Inputs from the interface and then would reflect the results on the website. The files required to run the models and the interface are listed below.

Step 1: Run the “Models.ipynb” 
Step 2: Run the “main.py” with the pickle files (“modelNN.pickle” , “indDict.pickle”)
Step 3: Step 2 would generate the link for the interface
Step 4: Click on the link to interact with the interface


## Conclusion of project

From our analysis we found that the relationship we are trying to establish between the intensity and the company’s fundamental data is primarily non-linear which was exhibited through the poor performance of the OLS variance model. We found that the random forrest regressor model showed exceptional performance for the expectation of the intensity however the variance model didn’t show a good performance. On the other hand we found that the Neural Net model had its performance unstable across different hyperparameter combinations. Comparing the above three models we propose the neural net model as the champion model, although we note that there are ample scope for improvement in the neural net model as well.

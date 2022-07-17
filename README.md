# SyriaTel Customer Churn Analysis

Author: Gamze Turan

<img align="center" width="600" height="300" src='images/people_phone.jpg'>

## Overview

I will examine the "SyriaTel Customer Churn" data in this study. The SyriaTel is a telecommunication company. To determine whether a customer will ("soon") discontinue doing business with Syria Tel is the goal of the study.

The best way the determine is to make a predictive model which will classify customers who might stop doing business with Syria Tel, using the data.

I will build a model for classifying whether customer will stop business True or False.

## Business Understanding

This search will detecting which customers are likely to leave a sevice or to cancel a subcription to a service.

Select a modelthat will be the most accurate in predicting which client will discontinue doing business with SyriaTel.

## Steps to solve project

1. Data Understanding
   * Loading Data
   * Data Overview
   * Data Preprocessing
2. Data Preparation
3. Modeling & Evaluation
4. Model 1 — Logistic Regression — SKlearn
5. Model 2 — Decision Tree Classifier
6. Model 3 — XGBoost + Grid Search
7. Evaluation
8. Pipelines for productionizing the model

## Data Understanding

### Loading Data 

I downloaded dataset "SyriaTel Customer Churn" from Kaggle that is used for observation.

The file name is 'bigml_59c28831336c6604c800002a.csv'.

Tha raw data has 3333 entries and 21 columns.

### Data Overview

- state 
- account length
- area code
- international plan
- voice mail plan
- number vmail messages
- total day minutes
- total day calls
- total day charge
- total eve minutes
- total eve calls
- total eve charge
- total night minutes
- total night calls
- total night charge
- total intl minutes
- total intl calls
- total intl charge
- customer service calls
- churn

After the observation of data we noticed that there are no missing values in the dataset.
Phone number columns is not important for modeling the data because it store random numbers
and we can't create dummy values that's why we will remove this columns.


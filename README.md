# Twitter-Sentiment-Analysis-NLP-Using-Python-Machine-Learning-

## Table of Contents

1. [Introduction](#introduction)

2. [Problem Statement](#problem-statement)

3. [Dataset Description](#dataset-description)

4. [Data Preprocessing](#data-preprocessing)

5. [Model Used](#model-used)

6. [Accuracy Metrics](#accuracy-score)

7. [Predictive Model](#predictive-model)
   
8. [Deployment](#deployment)

9. [Insights](#insights)

11. [Conclusion](#conclusion)

## 1. Introduction

This project focuses on analyzing sentiments in tweets using Natural Language Processing (NLP) techniques and a Logistic Regression model. The objective is to classify tweets as positive or negative on their content. By analyzing the sentiment of tweets, we aim to gain insights into public opinions on various topics.

## 2. Problem Statement

Social media platforms like Twitter generate vast amounts of textual data daily. Understanding public sentiment from this data is valuable for businesses, policymakers, and researchers. However, manually analyzing such a large dataset is impractical. This project provides an automated approach to classify tweet sentiments using machine learning.

## 3. Dataset Description

This is the sentiment140 dataset. It contains 1,600,000 tweets extracted using the twitter api . The tweets have been annotated (0 = negative, 1 = positive) and they can be used to detect sentiment .

The dataset consists of 6 columns, namely:

- target: the polarity of the tweet (0 = negative, 1 = positive)

- ids: The id of the tweet (2087)

- date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)

- flag: The query (lyx). If there is no query, then this value is NO_QUERY.

- user: the user that tweeted (robotickilldozr)

- text: the text of the tweet (Lyx is cool)

![image_alt](https://github.com/Shamiso-Tirivanhu/Twitter-Sentiment-Analysis-NLP-Using-Python-Machine-Learning-/blob/9f76e16feb090d780c2717ff43c197870ba3188e/Data%20description.png)


## 4. Data Preprocessing

- The dataset used in this project consists of 1,600, 000 tweets, sourced from Kaggle. Instead of manually downloading the dataset, we utilized the Kaggle API to load the dataset in JSON format for further analysis.

The following preprocessing steps were performed on the dataset:

- Lowercasing: Converted all text to lowercase to maintain uniformity.

- Removing Special Characters & URLs: Eliminated unnecessary symbols, punctuation, and links.

- Tokenization: Split text into individual words for better analysis.

- Stopword Removal: Removed common words that do not contribute to sentiment classification.

- Vectorization: Converted text data into numerical format using TF-IDF.

- The twitter sentiment dataset had no missing values.

- The dataset was evenly distributed into two parts. That is 0 = 800 000 & 1 = 800 000


## 5. Model Used

This project employed the Logistic Regression model for sentiment classification. Logistic Regression is a widely used algorithm for binary and multiclass classification tasks and provides a strong baseline for text classification problems. Henceforth,  Logistic Regression was deployed because we were working with binary classification i.e (0 = negative tweets, 1 = positive tweets)

## 6. Accuracy Metrics

The model's performance was evaluated using the accuracy score:

- Training Accuracy: 79.9%

- Testing Accuracy: 77.6%

Based on the abovementioned accuracy score, it can be denoted that the Logistic Regression model performed well. The accuracy score on testing data is closer to the one obtained on the training data. Essentially, our model can predict tweets with decent measure of alacrity. 


![image_alt]()


## 7. Predictive Model 

The dataset was evenly split into training and testing sets using train_test_split(), ensuring a balanced distribution of sentiments across both subsets.

![image_alt]()


## 8. Deployment

- To streamline future predictions, the trained model was saved using Pickle. This allows us to use the model without retraining it every time, significantly improving efficiency in real-world applications.

- A predictive system was also build to test if the model is working.


![image_alt]()


## 9. Insights

From analyzing the dataset, we observed:

- Specific words and hashtags contributed significantly to sentiment classification.

- The Logistic Regression model provided a good balance between accuracy and interpretability.

## 10. Conclusion

- Using Logistic Regression, we successfully classified tweets into positive & negative categories. 

- The project demonstrated the effectiveness of NLP techniques in analyzing social media sentiment. 

- Future work could involve exploring more advanced models such as deep learning techniques to further improve classification performance.

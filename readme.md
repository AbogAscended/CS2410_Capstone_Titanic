# CS 2410 Capstone Project - Kaggle Titanic Competition
## How to use repository
Simply clone the repository, all the data and everything you need including the separately defined data preprocessor
are all included. Simply open the Final.ipynb notebook and run all cells, and it will produce another submission.csv in
the submission folder for you to submit to kaggle if you so wish just know you are cheating. If you are interested I
also have an earlier created model with not as good of performance saved in the models folder.

## Goal/Motivation
In this project I am to analyze the data of the titanic competition on kaggle found here which is a classic kaggle competition designed to help people learn the workings of data science and machine learning. Through this project I plan to dive in depth in the relations between the different features(variables) within this data set as well as how to best replace missing data and finally then create a machine learning model to classify who did or did not survive the titanic, the ultimate goal of the actual competition.

## Data Discovery and Discussion:
Out of all values of the data set the only values missing are part of the ‘Age’ column. So to fix this problem I plan to use polynomial interpolation using the pandas inbuilt system/method. The only problem with this is choosing the correct order polynomial so I don't choose a too high order polynomial where I end up overfitting the data and not too low so that I don't generalize too hard and lose information. I initially chose to try and choose visually but It seemed to be a poor choice as I couldn't not tell the different points apart on the same or different plot. At the same time there does seem to be a few outliers in the dataset so removing the outlying two percent should be a good idea. So I will be removing everything outside three standard deviations from the data set. The next big thing I did was look at the correlation matrix which I generated using pandas df.corr() command and then visualized it using seaborn heatmap. I used this to see what numerical columns already had high correlations and which did not. I also used this to imagine what combinations I could do to possibly create more high correlation features. I then filled in the missing cabins by using the fact that pclass is already basically based on that concept at 1 defined people who would be in cabins A and B, 2 in C and D and so on. I filled the data and did some engineering on the names to make it so that each only retained their titles and then mapped that onto numbers manually ordinal encoding this feature. I also created a family size column and then divided fare by that number to create a fare per person column. After adding new features I then checked correlation again and dropped anything with a correlation much smaller than anything else.

## Model Design and Implementation:
For this part I used a gradient boosted forest classification from Xgboost in Sklearn. After doing my feature engineering I then implemented a basic optuna objective function which I used to find the best hyperparams for this model. To then actually measure the effectiveness of the model with its features and hyperparameters I did a ‘manual’ implementation of cross model validation using the StratifiedKFold method which splits the data to measure the performance between multiple models and averages that. This was the actual score used for the optuna hyperparam search which was set to maximize as the better the score the closer to perfect that actual model was. I used early stopping and lots of overfitting hyper params as through the entire fitting period that was the biggest problem as even with 10 split validation i was returning an average about .82 while the highest I could get the actual to be was .79904 earning me place 805/15280 on the titanic leaderboard as of november 26, 2024. Overall the libraries I used were pandas, optuna, numpy, seaborn, matplotlib, and Sklearn.

## Society/Community Impact:
This dataset doesn’t really have a community impact but instead I will talk about how this impacts me and the kaggle community. Kaggle is really interesting because at the same time as being a place of competition for money its also an interesting community where people can share ideas, code, notebooks, and models to tackle wide problems. It even has a large assortment of datasets not connected to any competitions at all. I think that kaggle is a great community to participate in extensively if you want to have a career in data science and will be nothing but helpful to you. This entire project basically all I wanted to do was break a certain number on the leaderboard and think about what others might have done to do better, and get a better score and also what I could do and have tried. I think that this makes data science interesting when you delve into weird and interesting different ways to engineer features and to try to do some weird combination or method to represent data to a machine learning model that other people have not done to achieve accuracy numbers not yet achieved. It's a fascinating field and I think that kaggle presents an opportunity to flex and grow those muscles at the same time. 

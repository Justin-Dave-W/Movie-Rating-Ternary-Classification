# Predicting Movie Performance Through IMDb Rating Classification


## General Overview
**Goal:**
- Create a classification model capable of predicting viewer's rating on movies based on a ternary categorical variable (satisfied/not satisfied)
- Identify prominent factors from the model that affects movie performance ratings
  
**Data Source:**
- https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset
- Shape = 5,043 rows and 28 columns
- Variable Examples: Director names, movie budget, content rating (PG-13, R), actor facebook likes, IMDb score, number of user reviews

**Models Tested:**
- Decision Tree
- Random Forest

**Model Optimization Methodologies:**
- Synthetic Minority Oversampling Technique (SMOTE)
- Hyperparameter Tuning using GridSearch

**Software Used:**
- Tableau
- Python
- MS Excel

## Table of Contents 
- [Data](#data)
- [Exploratory Analysis](#exploratory-analysis)
- [Cross Tabular Visualizations](#cross-tabular-visualizations)
- [Model Benchmarking and Optimization](#model-benchmarking-and-optimization)
- [Final Remarks](#final-remarks)


## Data
The dataset was obtained through the platform, Kaggle and holds a diverse movie record from 1927 to 2016. Some examples of the variables in this dataset are as follows:
- **Movie details**			        : Director name, release year, language, and country
- **Audience & Critic Metrics**	: IMDb scores, number of users and critic reviews
- **Financial Information**		  : Movie budget, gross revenue
- **Categorical Data**		      : Color format, content rating (R, PG-13)
- **Popularity Measures**		    : Actor 1 Facebook likes, Director Facebook likes


## Exploratory Analysis 

### Removing Outliers
The original shape of the data includes 5,043 rows and 28 columns. The data cleaning procedures we performed included removing outliers, duplicate records, and any rows with missing values. These three procedures dropped our dataset’s total records from 5,043 to 3,566. We used the IQR rule to identify outliers and evaluate if they are entry errors. Although most are not errors, some variables like movie budgets contain actual outliers. We noticed that foreign language movies have budgets of up to $12 billion dollars. We suspect these values are presented using foreign currencies, so we decided to remove any foreign language movie and their respective budget outliers, resulting in our budget histogram improving drastically. 

![image](https://github.com/user-attachments/assets/31cf2e68-6f96-47de-9856-5da3b53d01ff)
![image](https://github.com/user-attachments/assets/43c980e9-8751-4fcb-b486-91c44d637aa1)


### Transforming Dependent Variable from Continuous to Categorical Format 
After cleaning the data, we performed four procedures for data engineering. First procedure is variable consolidation on three categorical variables, where we took three categories with the highest proportions and stored the other categories in the “Others” group. Second procedure is one hot encoding, where we create dummy variables for all the categorical variables. However, we dropped 7 name columns like actor and director names, which contains too many unique values for one hot encoding. Third procedure is creating a correlation heatmap where we discovered a 0.95 positive correlation between actor_1_facebook_likes and cast_total_facebook_likes, which led us to delete the former variable. The final procedure is engineering our dependent variable called IMDb_score_category, where we transformed the numerical IMDb_score variable to a categorical movie performance tier list as shown in the table below. 

![image](https://github.com/user-attachments/assets/e6c901e3-9744-4ea4-b9b7-892c18a933ed)

![image](https://github.com/user-attachments/assets/fb3b2035-c7da-422d-8c66-552e6c508435)
![image](https://github.com/user-attachments/assets/2c2d55e0-1102-4773-9a84-75dc91500854)

The two graphs above show the distribution and proportion of the numerical IMDb Score variable and our target categorical variable, IMDb Score Category. As seen in the histogram, the distribution of the IMDb Score is nearly bell shaped, with most of the distribution falling between 5 and 8 ratings. Because of the nature of this distribution, the majority class in our ‘IMDb Score Category’ variable is mediocre (5 < IMDb score < 7.5). ‘Poor movies’ and ‘good movies’ classes represent only 8.1% and 15.4% of the target variable’s proportion. This might cause issues during the data modeling process as classification models tend to classify majority classes better than minority classes, leading to unbalanced models. 


## Cross Tabular Visualizations 
### Table 1 ~ Mean Facebook Likes per IMDb Category

![image](https://github.com/user-attachments/assets/233374e6-e044-4c40-baa1-e19d311c9a88)

The dataset quantifies popularity through Facebook likes. Table 1 above compares the average movie, total cast, and director facebook likes between each of the IMDb movie categories. Total cast facebook likes decreases gradually from good to poor movies. However, good movies have a significantly higher director and movie facebook likes than the other 2 categories, which potentially suggests that if an abnormally large number of people likes a movie post and a very popular director is directing the movie, that movie might be classified as a good movie. 

### Table 2 ~ Mean Gross Revenue and Budget per IMDb Category 

![image](https://github.com/user-attachments/assets/1e68c682-4ed3-48a8-9845-55c503e3dacd)

Table 2 above compares the average gross revenue earned and budget spent per IMDb category. Focusing on the revenue, good movies make significantly higher revenue than both mediocre and poor movies. In regards to budget, the amount spent by good and mediocre movies are nearly identical. However, poor movies invest on average $10 million USD less than the budget for good and mediocre movies. Hence, the amount of budget spent can potentially differentiate poor movies from mediocre and good ones.

### Table 3 ~ Mean Number of Critic Reviews and User Reviews & Votes per IMDb Category 
![image](https://github.com/user-attachments/assets/711c0039-4b7c-41e5-bbbf-bb66e9094747)
![image](https://github.com/user-attachments/assets/aafebed4-bda1-4386-b915-0fe2287d910c)

Table 3.1 and 3.2 above represents a comparison of the average number of user reviews, critic reviews, and user votes between the three IMDb categories. Reviews are the text feedback posted about a movie by either a normal user or a critic. Votes are a simple 1 to 10 IMDb rating score provided by the users. As seen in the two graphs, the amount of user reviews and user votes are significantly higher for good movies compared to mediocre and poor movies. This suggests that more people are willing to post an IMDb score rating and written review if the movie is good compared to mediocre and poor movies. Critic reviews are on average higher than good movies, but not at a significant difference compared to user reviews and votes. 


## Model Benchmarking and Optimization
The following models utilize cleaned and engineered data that has been partitioned into a 70% training set and 30% testing set.

### Decision Tree 
We chose the decision tree as our first classification model due to its interpretable nature, as each node displays the features used to divide the data into purer subsets. First, we created a benchmark model by simply feeding the data into sklearn’s DecisionTreeClassifier algorithm using entropy as the model’s criterion. The model was overfitted with a 100% training accuracy and 75.42% testing accuracy. Additionally, we noticed that the recall accuracy between the poor, mediocre, and good movies is extremely uneven as displayed in the table to the right. Recall is the proportion of actual positive classes that is classified correctly as positive. As we can see, the model is better at classifying mediocre movies with an 83% recall accuracy compared to poor and good movies with a 39% and 58% recall accuracy respectively. Hence this model has two problems, overfitting and imbalance recall accuracy.

![image](https://github.com/user-attachments/assets/0198679a-4dd6-4f9b-9ffe-1bdabb4be0e0)

#### Pre Pruning To Remove Overfitting 
To reduce overfitting, we performed pre pruning by limiting the growth of the decision tree by setting maximum and minimum values for the depth, sample splits, and sample leaf nodes. We utilized hyperparameter tuning using gridsearch to find the optimal values for the three parameters stated above. Three trials of gridsearch were performed with three different parameter combinations and the following best result was obtained. The Best Fit Model section shows the optimal parameters to build our decision tree. This model is no longer overfitted, with a training accuracy of 82.81% and a testing accuracy of 81.21%. Overall, this model is an upgrade compared to our benchmark model, however, the recall accuracy is still extremely imbalanced. The model has a 94% recall accuracy for mediocre movies and less than 50% accuracy for the other two categories. Next, we used SMOTE sampling to fix this imbalanced accuracy issue. 

![image](https://github.com/user-attachments/assets/5dc7e15b-28db-4b40-a952-eae356e14ead)

#### SMOTE Sampling to Remove Target Class Imbalance 
The root cause of the imbalanced recall accuracy is the dependent variable’s uneven class proportion. As mentioned in the exploratory analysis of this report, the mediocre movie class constitutes 76.5% of the target variable. Because of this, the model is better at predicting the dominant class compared to the minority classes, which are good and poor movies. To fix this issue, we tried using SMOTE sampling which evens out the class distribution of the dependent variable in the training dataset. On top of using the SMOTE data, we build a new decision tree model using hyperparameter tuning (Gridsearch) to prevent the model from overfitting. The following shows the result of this new model.

![image](https://github.com/user-attachments/assets/3f7c428b-bc12-4842-9943-bc8b09e8eb40)

The model is not overfitted, but the testing accuracy when using the SMOTE data decreased to 72.8% compared to the 81.21% accuracy when using the original data. However, the recall accuracy is more even in this new SMOTE + pre pruned model. Although the recall accuracy for the mediocre movie class decreased by 17%, the poor movie and good movie recall accuracy increased by around 20%. Overall, even after using SMOTE, the recall accuracy imbalance issue is still present. Next, we tried using the random forest model to create a better model to classify the IMDb categories. 


### Random Forest 
In an attempt to further improve the accuracy of our Decision Tree model, we used the ensemble learning algorithm of Random Forest. Before processing any of our data, we ran a benchmark model to observe how well the model will perform. We stayed consistent with our Decision Tree model by using entropy as the criterion. Our model produced a training accuracy of 100% and testing accuracy of 84.21%. The 100% training accuracy is a clear indication of an overfitted model showing us that certain pre-processing steps will need to be executed. Additionally, the recall accuracies of 5%, 98%, and 55% on poor, mediocre, and good movies show an imbalance learning model that is very bad as predicting poor and good movies.

![image](https://github.com/user-attachments/assets/59c80894-6c89-4f5a-8e9e-adb9601aecd6)


#### Pre Pruning To Remove Overfitting 
In order to account for our overfitted model, we used sklearn’s GridSearchCV to pre-prune each decision tree so the model has a better generalized performance. We limited the growth of max depth, max attributes used on each tree, minimum samples allowed per leaf node, and minimum samples needed to split a node. Similarly to our Decision Tree model, we ran multiple trials of grid search in order to find the optimal parameter values for our model. This resulted in a training accuracy of 84.98% and testing accuracy of 84.21%. By limiting certain parameters we solved for our overfitted model. However, the recall accuracy of this model still performed poorly thus we will use the SMOTE technique to improve the recall.


#### SMOTE and Pre Pruning Random Forest 

![image](https://github.com/user-attachments/assets/ee632b72-dc7a-4359-bf14-ec5acc124560)

Our final model combines the SMOTE technique and GridSearchCV in an attempt to solve for overfitting and improve recall accuracy. As stated in the Decision Tree section of this report, our dataset contains a dominant proportion of mediocre movies thus making the model strong at predicting mediocre and weak at predicting good or poor movies. SMOTE helps create a balanced dependent variable in the training set, thus solving for the issue. We also used multiple GridSearchCV trials to find the optimal values for the hyperparameters. This model solved for overfitting with a training accuracy of 95.32% and an overall accuracy of 90.14%, making it our most accurate model. By using the SMOTE sampling technique, we also managed to increase our recall accuracy to 94%, 85%, and 91% for poor, mediocre, and good movies respectively.

![image](https://github.com/user-attachments/assets/8d86b899-efd8-4219-9e02-0d329fafbbce)


## Final Remarks
Since the random forest model outperforms the decision tree model in regards to testing and recall accuracy, we will gather and discuss insights based on the random forest model. The following table shows the top 12 features used in the random forest model.

![image](https://github.com/user-attachments/assets/a6b9c847-0598-46f6-b102-9ccf3641cf36)

The number of user votes and reviews are the top two most important features in the model. Referring back to table 3.1 and 3.2, good movies typically have significantly higher votes and reviews than mediocre and bad movies. Additionally, movie and director facebook likes are also the top 5 and 11 variables. Referring back to table 1, the number of facebook likes for the movie and directors are significantly higher for good movies than the other two categories. Hence, if a movie studio or regular people want to predict if a movie is going to be good or not good (mediocre/poor), they should check the number of reviews and ratings left by normal users through IMDb and the popularity of the movie and directors in social media. Additionally, the budget variable is ranked in the top 6 most important features. As shown in table 2, movies with higher budget tend to be either good or mediocre movies. According to a study in Nitte School of Management, movies with higher budgets are associated with higher audience ratings (Prasad, 2023).

Returning to the question from the report’s related work section, it appears that the results of Filmgrail’s sentiment analysis aligns with our model’s results. The intensity of the audience’s reaction does hold significant predictive power in determining movie performance. In our model, audience reaction is represented by the normal users (audience) reviews and votes as well as people’s Facebook likes for the movie itself. We would recommend the movie studios to also observe the popularity of the directors, as more popular directors tend to be associated with good movies as seen in Table 1. Although higher movie budgets don’t necessarily lead to good movies exclusively, this variable is shown to separate good and mediocre movies from poor movies. These added insights should benefit the studio in making their marketing budget allocation decisions.

We understand that some of these variables, like budget and revenue, aren't made available before the movie’s release. However, we can recommend normal viewers to research on the popularity of the director and how much ‘hype’ the movie has through social media before the movie’s release to validate the movie’s performance. To make the classification more accurate, viewers can also check the number of votes and reviews left by normal users slightly after the movie’s release. Instead of focusing simply on the IMDb rating number (1-10) which most people do these days, this study has shown the importance of observing the number of people who left the rating and reviews. 
















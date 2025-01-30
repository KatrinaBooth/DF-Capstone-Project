<a id="readme-top"></a>

# Workout Recommender and Calorie Predictor

## Digital Futures Academy Capstone Project

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#tools-used">Tools Used</a>
    </li>
    <li>
      <a href="#deliverables">Deliverables</a>
    </li>
    <li>
      <a href="#outcomes">Outcomes</a>
    </li>
    <li>
      <a href="#conclusion">Conclusion</a>
    </li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This is my final project of the Digital Futures Academy, showcasing some of the skills I've learnt throughout. I decided to choose a project based on a topic I'm really passionate about - fitness. I found a dataset containing (supposedly) the best 50 Exercises for your body on Kaggle which contains the exercises name, the number of sets and reps, the benefits, the number of it calories burns, which muscles it targets, the difficulty level and whether equipment is needed or not. 

This 1-week project focuses on creating a user interface using Streamlit that can recommend personalized workout exercises and predict the number of calories burned during workouts. The recommender system uses user preferences, such as target muscle groups, exercise difficulty, and equipment availability, to filter a dataset of exercises and suggest the most suitable options based on their calorie-burning potential. In parallel, a calorie prediction model was developed using features like exercise duration, intensity, and body metrics to estimate calorie expenditure. I started with a train-test split of the data and exploratory data analysis to identify key patterns and correlations. Following this, I implemented feature engineering, included one-hot encoding categorical variables, creating new features like total reps, and scaling numerical features using MinMaxScaler to ensure consistency. I tested various models, but linear regression using L2 regularization was selected for its balance between interpretability and performance, which was assessed using metrics like R², RMSE, and residual analysis to validate its predictive accuracy. To incorporate pysical characteristics into the final predictor, I found a second dataset containing atrributes such as sex, gender, height, ... and calories burned. I used this to calculate the average percentage increase in calories burned between genders, different age categories, and different BMI categories, keeping relevant confounders constant (exercise intensity, sex, age and BMI). This was then used to find a scale factor which could be applied to the output of my calorie prediction model to futher tailor the result.

By employing these techniques, the project aims to provide a reliable and efficient system for optimizing workout plans and helping users track their fitness progress.


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- Tools Used -->
## Tools Used

* **Python**: For exploratory data analysis, feature engineering and modelling.
* **Jupyter Lab**: IDE for the project.
* **Libraries**:
  * `numpy`
  * `pandas`
  * `matplotlib`
  * `seaborn`
  * `sklearn`
  * `statsmodels`
  * `pickle`
* **Kaggle**: For sourcing the data.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- Deliverables -->
## Deliverables

* A workout recommender which maximises the calorie burn column based on user inputs (their skill level, which body parts they'd like to train, whether they have access to equipment etc). 

* A calorie predictor which predicts the number of calories burnt for a completely new exercise based on the number of sets and reps they do, the muscle it targets and difficulty level. There's also the ability to input physical characteristics such as sex, age and BMI to further tailor the calorie prediction to the user.

* A user friendly interface combining these two tools using Streamlit.
  
* A 10 minute presentaion to be delivered to the cohort and Digital Futures employees.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- Outcomes -->
## Outcomes

* The R-Squrared statistic for my regularised linear regression model was 0.81.
* The RMSE on the train and test data for my model was 27 and 28 respectively.
* Homoscedasticity and normality assumptions of linear regression were met through residual analysis.
* Easy-to-use user interface which can recommend exercises for your workout and predict the number of calories burned given a new set of exercises.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- Conclusion -->
## Conclusion

This project effectively delivered a two-part tool aimed at improving workout personalization and performance analysis, demonstrating how predictive modelling can be used to help tackle the lack of consistent exercise present in the UK. Even though this project has been a success, it hasn't been without limitations. The main issue I faced was the lack of exercise data available in the dataset - only 50 rows. While this provides a good starting point, it limits the diversity of workout suggestions and the accuracy of calorie prediction. To address this, I tried to include oversampling but the techniques available to me were either meant for classification models or my dataset was still too small to implement them. Random oversampling was also considered and tested but it led to overfitting so I decided against it. To enhance this project in the future I would consider webscraping as a technique to expand my dataset as there were no other options on Kaggle and the limited time allocated to this project wasn't sufficient to pursue this option. 


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Katrina Booth: katrinalilybooth@gmail.com

Project Link: [https://github.com/KatrinaBooth/DF-Capstone-Project](https://github.com/KatrinaBooth/DF-Capstone-Project)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

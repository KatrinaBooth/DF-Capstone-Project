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

This project focuses on creating a user interface using Streamlit that can recommend personalized workout exercises and predict the number of calories burned during workouts. The recommender system uses user preferences, such as target muscle groups, exercise difficulty, and equipment availability, to filter a dataset of exercises and suggest the most suitable options based on their calorie-burning potential. In parallel, a calorie prediction model was developed using features like exercise duration, intensity, and body metrics to estimate calorie expenditure. By employing techniques such as exploratory data analysis (EDA), feature engineering, and model evaluation, the project aims to provide a reliable and efficient system for optimizing workout plans and helping users track their fitness progress.

The workout recommender was built by filtering the dataset based on user preferences and selecting the desired number of exercises, prioritizing those that burn the most calories. The calorie predictor required a lot more skill, starting with a train-test split of the data and exploratory data analysis to identify key patterns and correlations. Feature engineering included one-hot encoding categorical variables, creating new features like total reps, and scaling numerical features using MinMaxScaler to ensure consistency. I tested various models, but linear regression using L2 regularization was selected for its balance between interpretability and performance. The model was assessed using metrics like R², RMSE, and residual analysis to validate its predictive accuracy. To incorporate pysical characteristics into the final predictor, I found a second dataset containing atrributes such as sex, gender, height, weight, ... and calories burned. I used this to calculate the average percentage increase in calories burned between men and women, different age categories, and different BMI categories, keeping relevant confounders constant (difficulty, sex, age and BMI). This was then used to find a scale factor which could be applied to the output of my calorie prediction model to futher tailor the result.



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

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- Deliverables -->
## Deliverables

* A workout recommender which maximises the calorie burn column based on user inputs (their skill level, which body parts they'd like to train, whether they have access to equipment etc). 

* A calorie predictor which predicts the number of calories burnt for a completely new exercise based on the number of sets and reps they do, the muscle it targets and difficulty level. There's also the ability to input physical characteristics such as sex, age and BMI to further tailor the calorie prediction to the user.
  
* A 10 minute presentaion to be delivered to the cohort and Digital Futures employees.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- Outcomes -->
## Outcomes

**UPDATE EVERYTHING BELOW FOR THIS PROJECT**

* The R-Squrared statistic for the accurate and ethical model was 0.984 and 0.944 respectively.
* The RMSE on the test data for both models was 1.243 and 2.213 respectively (outperforming the given baseline).
* The Country feature was found to be far too powerful in predicting life expectancy so any prediction would be dependent on the country rather than any other inputted features. This raised ethical concerns for underperforming countries so this feature was removed from both models.
* Features relating to immunisation converage and disease incidence were removed from the ethical model because these features reflect directly on the healthcare infrastructure of the country which could put financial pressure on a country to increases funding.
* Features relating to thinness prevalence are a reflection on levels of nutrition across a country which is sensitive information. These features were therefore removed from the ethical model.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- Conclusion -->
## Conclusion

This project effectively delivered two predictive linear regression models and an interactive function for estimating life expectancy, prioritizing both accuracy and ethical standards. The accuracte model demonstrated superior predictive performance with a lower RMSE, while the ethical model provided a privacy-focused alternative. These results highlight the capability of data-driven approaches to deliver impactful, ethical, and scalable solutions in the realm of public health.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Katrina Booth: katrinalilybooth@gmail.com

Project Link: [https://github.com/KatrinaBooth/WHO-Life-Expectancy-Predictions](https://github.com/KatrinaBooth/WHO-Life-Expectancy-Predictions)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

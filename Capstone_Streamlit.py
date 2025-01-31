import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from sklearn.preprocessing import MinMaxScaler


# CSS to set the background image
st.markdown(
    f'''
    <style>
    .stApp {{
        background-image: url('https://static.vecteezy.com/system/resources/thumbnails/047/752/116/small/barbell-flat-icon-isolated-on-transparent-background-vector.jpg');
        background-size: cover;
        background-position: center center;
        background-attachment: fixed;
    }}
    </style>
    ''',
    unsafe_allow_html=True)


################## DATA MANIPULATION ##################

df = pd.read_csv('Top 50 Excerice for your body.csv')

# Add a URL to the exercise instructions
df['URL'] = ['https://musclewiki.com/bodyweight/male/chest/push-up', 'https://musclewiki.com/bodyweight/male/glutes/bodyweight-squat',
            'https://musclewiki.com/bodyweight/male/chest/push-up', 'https://musclewiki.com/bodyweight/male/abdominals/burpee',
            'https://musclewiki.com/bodyweight/male/abdominals/mountain-climber', 'https://musclewiki.com/cardio/male/gluteus-medius/cardio-jumping-jacks', 
            'https://musclewiki.com/bodyweight/male/abdominals/bicycle-crunch',
            'https://musclewiki.com/bodyweight/male/triceps/parralel-bar-dips', 'https://musclewiki.com/bodyweight/male/traps-middle/pull-ups',
            'https://musclewiki.com/bodyweight/male/abdominals/bodyweight-russian-twist', 'https://musclewiki.com/bodyweight/male/abdominals/laying-leg-raises',
            'https://musclewiki.com/barbell/male/traps-middle/barbell-deadlift', 'https://musclewiki.com/barbell/male/chest/barbell-bench-press',
            'https://musclewiki.com/dumbbells/male/traps-middle/dumbbell-row-unilateral', 'https://musclewiki.com/dumbbells/male/shoulders/dumbbell-seated-overhead-press',
            'https://musclewiki.com/bodyweight/male/calves/calf-raises', 'https://musclewiki.com/dumbbells/male/triceps/dumbbell-overhead-tricep-extension',
            'https://musclewiki.com/dumbbells/male/shoulders/dumbbell-lateral-raise', 'https://musclewiki.com/bodyweight/male/glutes/glute-bridge',
            'https://musclewiki.com/bodyweight/male/lowerback/supermans', 'https://musclewiki.com/bodyweight/male/glutes/box-jump',
            'https://musclewiki.com/kettlebells/male/lowerback/kettlebell-swing', 'https://musclewiki.com/kettlebells/male/glutes/kettlebell-step-up',
            'https://musclewiki.com/cables/male/shoulders/machine-face-pulls', 'https://musclewiki.com/machine/male/lats/machine-pulldown',
            'https://musclewiki.com/bodyweight/male/hamstrings/bodyweight-reverse-lunge', 'https://musclewiki.com/bodyweight/male/glutes/jump-squats',
            'https://musclewiki.com/bodyweight/male/abdominals/sideways-scissor-kick', 'https://musclewiki.com/bodyweight/male/triceps/bench-dips',
            'https://musclewiki.com/machine/male/biceps/machine-seated-cable-row', 'https://musclewiki.com/bodyweight/male/abdominals/scissor-kick',
            'https://musclewiki.com/bodyweight/male/traps-middle/inverted-row', 'https://musclewiki.com/bodyweight/male/glutes/bulgarian-split-squat',
            'https://www.nasm.org/workout-exercise-guidance/how-to-floor-prone-cobra?srsltid=AfmBOoqMy1G2C7rHKHBXV1skXTlWW6wIBz5GT_yD47CLKy44RP-T0Um9',
            'https://musclewiki.com/band/male/traps-middle/band-pull-apart', 'https://musclewiki.com/recovery/male/front-shoulders/wall-angels',
            'https://musclewiki.com/bodyweight/male/abdominals/bird-dog', 'https://musclewiki.com/bodyweight/male/triceps/bodyweight-explosive-push-up',
            'https://musclewiki.com/bodyweight/male/triceps/decline-push-up', 'https://musclewiki.com/bodyweight/male/triceps/incline-push-up',
            'https://musclewiki.com/bodyweight/male/abdominals/dead-bug', 'https://musclewiki.com/bodyweight/male/glutes/bodyweight-single-leg-squat',
            'https://musclewiki.com/dumbbells/male/biceps/dumbbell-incline-zottman-curl', 'https://greatist.com/fitness/dragon-flag#how-to',
            'https://musclewiki.com/dumbbells/male/abdominals/dumbbell-renegade-row', 'https://www.setforset.com/blogs/news/frog-jumps-exercise?srsltid=AfmBOoradmLwDMH8XxR7h8R2QI9GhDaLs5Hj2dRii4duL2ixiU2EeU35',
            'https://www.self.com/story/how-to-do-a-turkish-get-up', 'https://www.verywellfit.com/how-to-do-the-bear-crawl-techniques-benefits-variations-4788337',
            'https://musclewiki.com/bodyweight/male/obliques/windshield-wiper', 'https://musclewiki.com/dumbbells/male/glutes/dumbbell-thruster']


# Map equipment to Yes or No
df.fillna(value = 'None', inplace = True)
df['Equipment'] = df['Equipment Needed'].apply(lambda x: 'No' if 'None' in x else 'Yes')

# Split the target muscle groups into individual rows and generalise the more specific muscle groups
df['Target Muscle Group'] = df['Target Muscle Group'].str.split(', ')
df = df.explode('Target Muscle Group')
df['Target Muscle Group'] = df['Target Muscle Group'].map({'Glutes':'Glutes', 'Core':'Core', 'Shoulders': 'Shoulders', 'Legs': 'Legs',
                                                           'Triceps': 'Triceps', 'Back': 'Back', 'Biceps': 'Biceps', 'Chest': 'Chest', 
                                                           'Full Body': 'Full Body', 'Forearms': 'Forearms', 'Calves': 'Calves', 
                                                           'Hips': 'Core', 'Full Core': 'Core', 'Lower Chest': 'Chest', 'Upper Chest': 'Chest', 
                                                           'Hip Flexors': 'Core', 'Obliques': 'Core', 'Lower Back': 'Back', 'Lower Abs': 'Core', 
                                                           'Rear Deltoids': 'Back', 'Upper Back': 'Back', 'Quadriceps': 'Legs', 'Hamstrings': 'Legs'})
df.drop_duplicates(inplace = True)




################## FUNCTIONS ##################

def scaling(df, scaled_col_names):
    df = df.copy()
    
    # List the features to be scaled
    features = df[scaled_col_names]
    
    # Fit and transform the scaler on the features to be scaled
    scaler = MinMaxScaler().fit(features)
    scaled_features = scaler.transform(features)
    df[scaled_col_names] = scaled_features
    return df



def feature_eng(df):
    df = df.copy()
    
    # Map equipment to yes - 1 or no - 0
    df.fillna(value = 'None', inplace = True)
    df['Equipment Needed'] = df['Equipment Needed'].apply(lambda x: 0 if 'None' in x else 1)

    # OHE the difficulty level - this performs better than mapping to 0, 1, 2.
    df = pd.get_dummies(df, columns = ['Difficulty Level'], drop_first = True, prefix = 'Difficulty', dtype = int)

    # Split the target muscle groups into individual rows, generalise the more specific muscle groups and keep the first row (most targeted muscle)
    df['Target Muscle Group'] = df['Target Muscle Group'].str.split(', ')
    df = df.explode('Target Muscle Group')
    df['Target Muscle Group'] = df['Target Muscle Group'].map({'Glutes':'Glutes', 'Core':'Core', 'Shoulders': 'Shoulders', 'Legs': 'Legs',
                                                           'Triceps': 'Triceps', 'Back': 'Back', 'Biceps': 'Biceps', 'Chest': 'Chest', 
                                                           'Full Body': 'Full Body', 'Forearms': 'Forearms', 'Calves': 'Calves', 
                                                           'Hips': 'Core', 'Full Core': 'Core', 'Lower Chest': 'Chest', 'Upper Chest': 'Chest', 
                                                           'Hip Flexors': 'Core', 'Obliques': 'Core', 'Lower Back': 'Back', 'Lower Abs': 'Core', 
                                                           'Rear Deltoids': 'Back', 'Upper Back': 'Back', 'Quadriceps': 'Legs', 'Hamstrings': 'Legs'})
    df = df.drop_duplicates(subset='Name of Exercise', keep='first')
    df = pd.get_dummies(df, columns = ['Target Muscle Group'], drop_first = True, prefix = 'Target', dtype = int)

    # Find the total number of reps for the exercise
    df['Total Reps'] = df.apply(lambda row: row['Sets'] * row['Reps'], axis = 1)

    # Scale the numeric columns
    df = scaling(df, df.select_dtypes(include=np.number).columns.tolist())
    
    return df




################## WORKOUT RECOMMENDER ##################

st.title('Workout Recommender')
st.write('This tool recommends personalised exercises that are most effective for burning calories, ensuring you get the most out of your workout routine!')

# Equipment / no equipment
equipment = st.selectbox('Do you have access to gym equipment?', options = df['Equipment'].unique())

# Difficulty level
difficulty_level = st.selectbox('Choose the maximum intensity level of your workout', options = ['Beginner', 'Intermediate', 'Advanced'])

# Target area
muscle_group = st.selectbox('Choose the muscle group you\'d like to work out', options = df['Target Muscle Group'].unique())

# Number of exercises
num_exercises = st.number_input('How many exercises would you like?', min_value = 1, step = 1)
    

# Filtering based on the user inputs
if equipment == 'No':
    if difficulty_level == 'Beginner':
        filtered_df = df[(df['Equipment'] == equipment) & 
        (df['Difficulty Level'] == difficulty_level) & 
        (df['Target Muscle Group'] == muscle_group)]
    elif difficulty_level == 'Intermediate':
        filtered_df = df[(df['Equipment'] == equipment) & 
        ((df['Difficulty Level'] == difficulty_level) | (df['Difficulty Level'] == 'Beginner')) & 
        (df['Target Muscle Group'] == muscle_group)]
    else:
        filtered_df = df[(df['Equipment'] == equipment) & 
        ((df['Difficulty Level'] == difficulty_level) | (df['Difficulty Level'] == 'Intermediate') | (df['Difficulty Level'] == 'Beginner')) & 
        (df['Target Muscle Group'] == muscle_group)]
else:
    if difficulty_level == 'Beginner':
        filtered_df = df[(df['Difficulty Level'] == difficulty_level) & 
        (df['Target Muscle Group'] == muscle_group)]
    elif difficulty_level == 'Intermediate':
        filtered_df = df[((df['Difficulty Level'] == difficulty_level) | (df['Difficulty Level'] == 'Beginner')) & 
        (df['Target Muscle Group'] == muscle_group)]
    else:
        filtered_df = df[((df['Difficulty Level'] == difficulty_level) | (df['Difficulty Level'] == 'Intermediate') | (df['Difficulty Level'] == 'Beginner')) & 
        (df['Target Muscle Group'] == muscle_group)]

# Sort by calories burned
filtered_df = filtered_df.sort_values(by = 'Burns Calories (per 30 min)', ascending = False)

# Limit the number of exercises
filtered_df = filtered_df[['Name of Exercise', 'Difficulty Level', 'Equipment', 'Equipment Needed', 'Sets', 'Reps', 'Benefit', 'Burns Calories (per 30 min)', 'URL']].head(num_exercises)

# Display Results
if not filtered_df.empty and filtered_df.shape[0] == num_exercises:
    st.write('Here are the exercises that match your criteria:')
    st.table(filtered_df)
elif not filtered_df.empty:
    st.write(f'There aren\'t {num_exercises} exercises that meet your criteria. Please try adjusting the filters to see more.')
    st.table(filtered_df)
else:
    st.write('No exercises match your criteria. Please try adjusting the filters.')




################## CALORIE PREDICTOR ##################

st.title('Calorie Predictor')
st.write('This tool predicts the total number of calories burned in your workout!')

# Initialize session_state for exercises if it doesn't exist
if 'exercise_info' not in st.session_state:
    st.session_state.exercise_info = {}

# Initialize total calories burned if not already in session_state
if 'total_calories' not in st.session_state:
    st.session_state.total_calories = 0


# Use these characteristics for scaling of the final prediction
gender = st.selectbox(f'What\'s your gender?', options=['Female', 'Male'])
age = st.selectbox(f'What\'s your age?', options=['20-29', '30-39', '40-49', '50-59', '60-69', '70-79'])
bmi = st.selectbox(f'What\'s your BMI?', options=['19-21', '22-24', '25-27', '28-30'])

# Scale this prediction based on percentage changes from the physical characteristics notebook
# Assume the exercise data is based off a 30 year old female with a BMI of 26.
gender_scaler = {'Female': 1, 'Male': 1.073975454163254275}
bmi_scaler = {'19-21': 1.02573484064793065, '22-24': 0.993833344, '25-27': 1, '28-30': 1.008523596527700462}
age_scaler = {'20-29': 0.9865682248, '30-39': 1, '40-49': 1.0016434036161519347, '50-59': 0.9788007236, '60-69': 0.9415439785, '70-79': 0.9117904178}



num_exercises = st.number_input("How many different exercises did you do in your workout?", min_value=1, step=1)

for i in range(num_exercises):
    st.write(f"Exercise {i + 1}")
    
    # Initialize the exercise entry in session_state if it doesn't exist
    if i not in st.session_state.exercise_info:
        st.session_state.exercise_info[i] = {'name': '', 'time': 1, 'calories': 0, 'additional_info': False, 'submitted': False}
    
    # Ask for the exercise name and sets inside the form
    with st.form(key=f"form_{i}"):
        name = st.text_input(f'What\'s Exercise {i+1} called?', key=f'name_{i}', value=st.session_state.exercise_info[i]['name'])
        time = st.number_input(f'How many minutes did you do Exercise {i+1} for?', min_value=1, max_value=60, step=1, key=f'time_{i}', value=st.session_state.exercise_info[i]['time'])
        
        # Create a submit button for the form
        submit_button = st.form_submit_button(label=f"Submit Exercise {i+1}")
    
    # Check if form was submitted
    if submit_button:
        # Store the current values in session_state
        st.session_state.exercise_info[i]['name'] = name
        st.session_state.exercise_info[i]['time'] = time
        st.session_state.exercise_info[i]['submitted'] = True
        
        # Check if the exercise name exists in the dataset
        if name and name.lower() in df['Name of Exercise'].str.lower().values:
            # Exercise found in the dataset
            calories = df[df['Name of Exercise'].str.lower() == name.lower()]['Burns Calories (per 30 min)'].values[0]
            scaled_calories = calories * gender_scaler[gender] * bmi_scaler[bmi] * age_scaler[age]
            cal = scaled_calories * time / 30
            st.session_state.exercise_info[i]['calories'] = cal
            st.write(f'Well done! You just burned {round(cal)} calories for Exercise {i+1}!')
            st.session_state.total_calories += cal
        else:
            # Exercise not found in the dataset, ask for additional info
            st.session_state.exercise_info[i]['additional_info'] = True

    # If additional information is required (exercise not in dataset), show the extra questions
    if st.session_state.exercise_info[i]['additional_info']:
        with st.form(key=f"form_details_{i}"):
            sets = st.number_input(f'How many sets did you do for Exercise {i+1}?', min_value=3, max_value=4, step=1, key=f'sets_{i}')
            reps = st.number_input(f'How many reps did you do per set for Exercise {i+1}?', min_value=5, max_value=20, step=1, key=f'reps_{i}')
            difficulty = st.selectbox(f'What\'s the difficulty of Exercise {i+1}?', options=['Beginner', 'Intermediate', 'Advanced'], key=f'difficulty_{i}')
            muscle = st.selectbox(f'What muscle group does Exercise {i+1} target the most?', options=['Core', 'Back', 'Legs', 'Chest', 'Full Body', 'Triceps', 'Shoulders', 'Glutes', 'Calves', 'Biceps'], key=f'muscle_{i}')
            
            # Submit button for the details form
            submit_details_button = st.form_submit_button(label=f"Submit Details for Exercise {i+1}")
        
        # If details were submitted, process the prediction
        if submit_details_button:
            # Store the additional details in session_state
            st.session_state.exercise_info[i]['sets'] = sets
            st.session_state.exercise_info[i]['reps'] = reps
            st.session_state.exercise_info[i]['difficulty'] = difficulty
            st.session_state.exercise_info[i]['muscle'] = muscle

            # Access the linear regression model from GitHub
            linreg_url = 'https://github.com/KatrinaBooth/DF-Capstone-Project/raw/refs/heads/main/LinRegModel.sav'
            response = requests.get(linreg_url)
            response.raise_for_status()
            model = pickle.loads(response.content)

            # Use the entered inputs to predict the calories burned
            user_input = pd.DataFrame([[name, sets, reps, difficulty, muscle, 'None']],
                                      columns=['Name of Exercise', 'Sets', 'Reps', 'Difficulty Level', 'Target Muscle Group', 'Equipment Needed'])
            user_input_feat_eng = feature_eng(user_input)

            feature_cols = ['Sets', 'Total Reps', 'Difficulty_Beginner', 'Target_Biceps', 'Target_Full Body', 'Target_Glutes', 'Target_Legs', 'Target_Shoulders', 'Target_Triceps']

            # Ensure all columns required for the model are there, fill with 0 if not
            missing_cols = set(feature_cols) - set(user_input_feat_eng.columns)
            for col in missing_cols:
                user_input_feat_eng[col] = 0

            prediction = model.predict(user_input_feat_eng[feature_cols])

            scaled_calories = prediction[0] * gender_scaler[gender] * bmi_scaler[bmi] * age_scaler[age]
            
            pred = scaled_calories * time / 30
            st.session_state.exercise_info[i]['calories'] = pred
            st.write(f'Well done! You just burned {round(pred)} calories for Exercise {i+1}!')
            st.session_state.total_calories += pred

# Display the total calories burned for the entire workout
st.write(f"Total calories burned for your workout: {round(st.session_state.total_calories)} calories!")






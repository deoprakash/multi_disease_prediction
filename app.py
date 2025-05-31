from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory, session, flash
import pickle
from flask_cors import CORS
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import pandas as pd
import joblib
from groq import Groq
import re
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, ValidationError
import bcrypt
from  flask_mysqldb import MySQL
import cloudinary.uploader
import cloudinary
from datetime import datetime
from dotenv import load_dotenv
# from flask_wtf.csrf import CSRFProtect
from datetime import timedelta
import json

load_dotenv()
app = Flask(__name__)

app.secret_key = os.getenv('SECRET_KEY')
CORS(app)
# csrf = CSRFProtect(app)
app.permanent_session_lifetime = timedelta(minutes=10)

# MySQL Config 
# app.config['MYSQL_HOST'] = 'localhost'
# app.config['MYSQL_USER'] = 'root'
# app.config['MYSQL_PASSWORD'] = 'deoprakash'
# app.config['MYSQL_DB'] ='user_database'

# mysql = MySQL(app)

# # Cloudinary Config
# cloudinary.config(
#     cloud_name="doiglu8td",
#     api_key="696122662926481",
#     api_secret="VUfY5PyHofVoufYl0zzDfan9MVQ",
#     secure=True
# )

# MySQL Config
app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST')
app.config['MYSQL_USER'] = os.getenv('MYSQL_USER')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD')
app.config['MYSQL_DB'] = os.getenv('MYSQL_DB' )
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_COOKIE_SECURE'] = True      # If using HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# app.secret_key = os.urandom(24)

mysql = MySQL(app)

# Cloudinary Config
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET'),
    secure=True
)


@app.before_request
def force_clear_session():
    session.permanent = True
    now = datetime.utcnow()

    if not session.get('session_initialized'):
        session.clear()
        session['session_initialized'] = True
        session['last_active'] = str(now)
    else:
        last_active = session.get('last_active')
        if last_active:
            try:
                last_active_dt = datetime.strptime(last_active, '%Y-%m-%d %H:%M:%S.%f')
                if now - last_active_dt > timedelta(minutes=5):
                    session.clear()
                    session['session_initialized'] = True
            except Exception:
                session.clear()
                session['session_initialized'] = True
        session['last_active'] = str(now)

# Registration Form
class RegisterForm(FlaskForm):
    name = StringField("Name", validators=[DataRequired()])
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Register")

    def validate_email(self, field):
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE email=%s", (field.data,))
        user = cursor.fetchone()
        cursor.close()
        if user:
            raise ValidationError('Email already exists')

# Login Form
class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Login")

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()
        cursor.close()

        if user and bcrypt.checkpw(password.encode('utf-8'), user[3].encode('utf-8')):
            session.permanent = False
            session['user_id'] = user[0]      # user[0] = id
            session['username'] = user[1]     # user[1] = name
            return redirect(url_for('index'))
        else:
            flash("Login failed. Please check your email and password")
            return redirect(url_for('login'))

    return render_template('login.html', form=form)

# Register Route
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        name = form.name.data
        email = form.email.data
        password = form.password.data

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        cursor = mysql.connection.cursor()
        cursor.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
                       (name, email, hashed_password.decode('utf-8')))
        mysql.connection.commit()
        cursor.close()

        flash("Registration successful! Please log in.", "success")
        return redirect(url_for('login'))

    return render_template('register.html', form=form)

# Logout Route
@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.clear()
    if request.method == 'POST':
        return '', 204  # No content for JS auto-logout
    return redirect(url_for('index'))


# Upload History
@app.route('/history')
def history():
    cursor = mysql.connection.cursor()

    user_id = session.get('user_id')

    # 1. Fetch all image-based predictions (e.g., cancer, brain_tumor, etc.)
    cursor.execute("""
        SELECT file_type, cloud_url, prediction, uploaded_at 
        FROM uploads 
        WHERE user_id = %s
    """, (user_id,))
    image_records = cursor.fetchall()

    # 2. Fetch all parameter-based predictions (e.g., cardio, diabetes, etc.)
    cursor.execute("""
        SELECT model_name, prediction, parameters_json, uploaded_at 
        FROM parameters_input 
        WHERE user_id = %s
    """, (user_id,))
    param_records = cursor.fetchall()

    entries = []

    # Append image-based records
    for file_type, cloud_url, prediction, uploaded_at in image_records:
        entries.append({
            'file_type': file_type,
            'prediction': prediction,
            'uploaded_at': uploaded_at,
            'cloud_url': cloud_url,
            'parameters': None  # No params for image-based
        })

    # Append parameter-based records
    for model_name, prediction, parameters_json, uploaded_at in param_records:
        try:
            parameters = json.loads(parameters_json) if parameters_json else {}
        except Exception:
            parameters = {}
        entries.append({
            'file_type': model_name,
            'prediction': prediction,
            'uploaded_at': uploaded_at,
            'cloud_url': None,  # No image for param-based
            'parameters': parameters
        })

    # Sort all entries by uploaded_at (most recent first)
    entries = sorted(entries, key=lambda x: x['uploaded_at'], reverse=True)

    cursor.close()

    return render_template("history.html", entries=entries)



cancer_model = pickle.load(open('Models/Cancer.pkl', 'rb'))
cardio_model = pickle.load(open('Models/cardiovascular.pkl', 'rb'))
iron_model = pickle.load(open('Models/IronDeficiency.pkl', 'rb'))
obesity_model = joblib.load('Models/obesity.pkl')
diabetes_model = joblib.load('Models/diabetes_model.pkl')
sicklecell_model = pickle.load(open('Models/sickle_cell_anemia.pkl', 'rb'))
brain_tumor_model = load_model('Models/brain_tumor_model.h5')
ckd_model = joblib.load("Models/decision_tree_ckd_model.pkl")
ckd_label_encoder = joblib.load("Models/ckd_label_encoder.pkl")
ckd_imp_features = joblib.load("Models/ckd_important_features.pkl")  # ['hemo', 'sg', 'sc', 'htn', 'sod', 'bgr']
common_disease_model = pickle.load(open("Models/common_disease_model.pkl", 'rb'))
common_disease_label_encoder = pickle.load(open("Models/common_disease_label_encoder.pkl", 'rb'))

@app.route('/')
def index():
    username = session.get('username')
    return render_template('index.html', username=username)

@app.route('/aboutUs')
def aboutUs():
    return render_template('aboutUs.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/select_model', methods=['POST'])
def select_model():
    model = request.form['model']
    if model == 'cardio':
        return redirect(url_for('cardio_form'))
    elif model == 'cancer':
        return redirect(url_for('cancer_form'))
    elif model == 'iron':
        return redirect(url_for('iron_form'))
    elif model == 'obesity':
        return redirect(url_for('obesity_form'))
    elif model == 'sicklecell':
        return redirect(url_for('sicklecell_form'))
    elif model == 'diabetes':
        return redirect(url_for('diabetes_form'))
    elif model == 'brain_tumor':
        return redirect(url_for('brain_tumor_form'))
    elif model == 'ckd':
        return redirect(url_for('ckd_form'))
    elif model == 'common_disease':
        return redirect(url_for('common_disease_form'))
    else:
        return "Invalid model selected", 400


# Class labels
anemia_class_names = ['ConjunctivaAnemia', 'FingerAnemia', 'NonAnemia', 'PalmAnemia']
cancer_class_names = ['ALL', 'Brain Cancer', 'Breast Cancer', 'Cervical Cancer', 'Healthy', 'Kidney Cancer', 'Lung and Colon Cancer', 'Lymphoma', 'Oral Cancer']
cancer_dir = 'static/cancer_dir'
iron_dir = 'static/iron_dir'
brain_tumor_dir = 'static/brain_tumor_dir'
bt_class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']


@app.route('/cancer', methods=['GET', 'POST'])
def cancer_form():
    if not os.path.exists(cancer_dir):
        os.makedirs(cancer_dir)

    if request.method == 'POST':
        if 'my_image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['my_image']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Generate timestamp and new filename
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        new_filename = f"pred_{timestamp}.jpg"
        public_id = f"pred_{timestamp}"
        folder = "cancer_predictions"

        # Save uploaded file with new name
        new_file = os.path.join(cancer_dir, new_filename)
        file.save(new_file)

        # Perform prediction
        prediction = predict_custom_image(cancer_model, new_file, cancer_class_names)

        # Upload to Cloudinary
        result = cloudinary.uploader.upload(new_file, public_id=public_id, folder=folder)
        cloud_url = result['secure_url']
        print("Cloudinary URL:", cloud_url)

        # Save to MySQL
        cursor = mysql.connection.cursor()
        user_id = session.get('user_id')
        cursor.execute("""
            INSERT INTO uploads (user_id, filename, file_type, cloud_url, prediction, uploaded_at)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (user_id, file.filename, 'cancer', cloud_url, prediction, datetime.now()))
        mysql.connection.commit()
        cursor.close()

        # Store cloud URL and prediction in session
        session['uploaded_cancer_img_url'] = cloud_url
        session['cancer_prediction'] = prediction

        return redirect(url_for('cancer_form'))  # Redirect to avoid resubmission

    # On GET request, retrieve from session
    cloud_url = session.get('uploaded_cancer_img_url')
    prediction = session.get('cancer_prediction')

    return render_template("cancer.html", prediction=prediction, img_url=cloud_url)

# Function to process and predict an image
def predict_custom_image(model, img_path, class_names):
    img = image.load_img(img_path, target_size=(128,128))  # Resize image
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = cancer_model.predict(img_array)
    return cancer_class_names[np.argmax(prediction)]  # Get class with highest probability

# Serve the uploaded image from the static directory
@app.route('/static/cancer_dir/<filename>')
def serve_cancer_image(filename):
    return send_from_directory(cancer_dir, filename)

@app.route('/cardio', methods=['GET', 'POST'])
def cardio_form():
    if request.method == 'POST':
        try:
            # Get form inputs safely
            gender = float(request.form.get('gender', 0))
            age = float(request.form.get('age', 0))
            cigsPerDay = float(request.form.get('cigsPerDay', 0))
            BPMeds = float(request.form.get('BPMeds', 0))
            diabetes = float(request.form.get('diabetes', 0))
            totChol = float(request.form.get('totChol', 0))
            sysBP = float(request.form.get('sysBP', 0))
            diaBP = float(request.form.get('diaBP', 0))
            BMI = float(request.form.get('BMI', 0))
            heartRate = float(request.form.get('heartRate', 0))
            glucose = float(request.form.get('glucose', 0))

            # Validate binary values
            if not (0 <= gender <= 1) or not (0 <= BPMeds <= 1) or not (0 <= diabetes <= 1):
                raise ValueError("Invalid binary field value (must be 0 or 1)")

            # Validate no negatives
            if any(val < 0 for val in [age, cigsPerDay, totChol, sysBP, diaBP, BMI, heartRate, glucose]):
                raise ValueError("Negative values are not allowed for numerical fields")

            # Create input array
            input_data = np.array([[gender, age, cigsPerDay, BPMeds, diabetes,
                                    totChol, sysBP, diaBP, BMI, heartRate, glucose]])
            
            # Ensure correct feature length
            if input_data.shape[1] != cardio_model.n_features_in_:
                raise ValueError(f"Model expects {cardio_model.n_features_in_} features, got {input_data.shape[1]}")

            # Perform prediction
            prediction = cardio_model.predict(input_data)
            prediction_label = "High Risk" if prediction.ravel()[0] == 1 else "Low Risk"

            # Convert inputs to JSON
            params_json = json.dumps({
                "gender": gender,
                "age": age,
                "cigsPerDay": cigsPerDay,
                "BPMeds": BPMeds,
                "diabetes": diabetes,
                "totChol": totChol,
                "sysBP": sysBP,
                "diaBP": diaBP,
                "BMI": BMI,
                "heartRate": heartRate,
                "glucose": glucose
            })

            # Save to database
            cursor = mysql.connection.cursor()
            user_id = session.get('user_id')
            cursor.execute("""
                INSERT INTO parameters_input (user_id, model_name, prediction, parameters_json, uploaded_at)
                VALUES (%s, %s, %s, %s, %s)
            """, (user_id, 'cardio', prediction_label, params_json, datetime.now()))
            mysql.connection.commit()
            cursor.close()

            return render_template('cardio.html', prediction=prediction_label)
        
        except ValueError as ve:
            return render_template('cardio.html', prediction=f"Input Error: {str(ve)}")
        except Exception as e:
            return render_template('cardio.html', prediction=f"Error: {str(e)}")

    return render_template('cardio.html', prediction=None)

anemia_class_names = ['ConjunctivaAnemia', 'FingerAnemia', 'NonAnemia', 'PalmAnemia']
    
@app.route('/iron', methods=['GET', 'POST'])
def iron_form():
    if not os.path.exists(iron_dir):
        os.makedirs(iron_dir)

    if request.method == 'POST':
        if 'my_image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['my_image']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Generate timestamp and new filename
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        new_filename = f"iron_{timestamp}.jpg"
        public_id = f"iron_{timestamp}"
        folder = "iron_predictions"

        # Save uploaded file with new name
        new_file = os.path.join(iron_dir, new_filename)
        file.save(new_file)

        # Perform prediction
        prediction = iron_custom_image(iron_model, new_file, anemia_class_names)

        # Upload to Cloudinary
        result = cloudinary.uploader.upload(new_file, public_id=public_id, folder=folder)
        cloud_url = result['secure_url']
        print(cloud_url)

        # Save to MySQL
        cursor = mysql.connection.cursor()
        user_id = session.get('user_id')
        cursor.execute("""
            INSERT INTO uploads (user_id, filename, file_type, cloud_url, prediction, uploaded_at)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (user_id, file.filename, 'iron', cloud_url, prediction, datetime.now()))
        mysql.connection.commit()
        cursor.close()

        # Store prediction and image URL in session
        session['uploaded_img_url'] = cloud_url
        session['prediction'] = prediction

        return redirect(url_for('iron_form'))  # Redirect to prevent resubmission

    # On GET: retrieve values from session
    cloud_url = session.get('uploaded_img_url')
    prediction = session.get('prediction')

    return render_template("iron.html", prediction=prediction, img_url=cloud_url)


def iron_custom_image(model, img_path, anemia_class_names):
    img = image.load_img(img_path, target_size=(240,240))  # Resize image
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = iron_model.predict(img_array)
    return anemia_class_names[np.argmax(prediction)]  # Get class with highest probability

@app.route('/static/iron_dir/<filename>')
def serve_anemia_image(filename):
    return send_from_directory(iron_dir, filename)


obesity_mapping = {'yes':1, 'no':0, 'Sometimes':1, 'Frequently':2, 'Always': 3, 'Walking': 0, 'Public_Transportation': 1, 'Motorbike': 2, 'Bike': 3, 'Automobile' : 4, 'Male':1, 'Female':0}
obesity_class = ['Insufficient_Weight','Normal_Weight', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III', 'Overweight_Level_I', 'Overweight_Level_II']


#Obesity

@app.route('/obesity', methods=['GET', 'POST'])
def obesity_form():
    if request.method == 'POST':
        try:
            # Get form values
            input_values = [
                request.form['gender'],
                request.form['age'],
                request.form['height'],
                request.form['weight'],
                request.form['family_history_with_overweight'],
                request.form['high-calorie_intake'],
                request.form['freq_vegetable_consumption'],
                request.form['main_meal_per_day'],
                request.form['food_between_meals'],
                request.form['smoke'],
                request.form['daily_water_intake'],
                request.form['calorie_intake'],
                request.form['phy_activity_freq'],
                request.form['time_using_tech'],
                request.form['alcohol_consumption'],
                request.form['transport_use']
            ]

            # Define correct feature names (must match training)
            feature_names = ['gender', 'age', 'height', 'weight', 'family_history_with_overweight',
                             'high-calorie_intake', 'freq_vegetable_consumption', 'main_meal_per_day',
                             'food_between_meals', 'smoke', 'daily_water_intake', 'calorie_intake',
                             'phy_activity_freq', 'time_using_tech', 'alcohol_consumption', 'transport_use']

            # Convert to DataFrame
            input_df = pd.DataFrame([input_values], columns=feature_names)

            # Apply the mapping
            input_df.replace(obesity_mapping, inplace=True)

            # Convert numeric columns to float/int as needed
            numeric_cols = ['age', 'height', 'weight', 'freq_vegetable_consumption',
                            'main_meal_per_day', 'daily_water_intake', 'phy_activity_freq', 'time_using_tech']
            input_df[numeric_cols] = input_df[numeric_cols].astype(float)

            # Predict
            prediction = obesity_model.predict(input_df)[0]
            result_label = obesity_class[int(prediction)] if isinstance(prediction, (int, np.integer)) else prediction

            # Convert inputs to JSON using input_df
            params_json = input_df.to_dict(orient='records')[0]  # Extract first row as a dictionary
            params_json = json.dumps(params_json)  # Convert to JSON string

            # Save to DB
            cursor = mysql.connection.cursor()
            user_id = session.get('user_id')
            cursor.execute("""
                INSERT INTO parameters_input (user_id, model_name, prediction, parameters_json, uploaded_at)
                VALUES (%s, %s, %s, %s, %s)""",
                (user_id, 'obesity', result_label, params_json, datetime.now()))
            mysql.connection.commit()
            cursor.close()

            return render_template('obesity.html', prediction=result_label)
        except Exception as e:
            return render_template('obesity.html', prediction=f"Error: {str(e)}")
    else:
        return render_template('obesity.html', prediction=None)

# Sickle Cell prediction

@app.route('/sicklecell', methods=['GET', 'POST'])
def sicklecell_form():
    if request.method == 'POST':
        try:
            input_values = [
                float(request.form['gender']),
                float(request.form['age']),
                float(request.form['hemoglobin']),
                float(request.form['red_blood_cell']),
                float(request.form['percent_blood_in rbc']),
                float(request.form['avg_vol_of_rbc']),
                float(request.form['avg_hg_per_rbc']),
                float(request.form['conc_hg_in_vol_of_rbc']),
            ]

            input_array = np.array(input_values).reshape(1, -1)
            prediction = sicklecell_model.predict(input_array)

            if prediction[0] == 1:
                result = "Anemia Detected"
            else:
                result = "No Anemia"
            
            # Convert inputs to JSON
            
            feature_names = ['gender', 'age', 'hemoglobin', 'red_blood_cell',
                         'percent_blood_in rbc', 'avg_vol_of_rbc',
                         'avg_hg_per_rbc', 'conc_hg_in_vol_of_rbc']
            input_dict = dict(zip(feature_names, input_values))
            params_json = json.dumps(input_dict)

            # Save to DB
            cursor = mysql.connection.cursor()
            user_id = session.get('user_id')
            cursor.execute("""
                INSERT INTO parameters_input (user_id, model_name, prediction, parameters_json, uploaded_at)
                VALUES (%s, %s, %s, %s, %s)""",
                (user_id, 'sicklecell', result, params_json, datetime.now()))
            mysql.connection.commit()
            cursor.close()

            return render_template('sicklecell.html', prediction=result)

        except Exception as e:
            return render_template('sicklecell.html', prediction=f"Error: {str(e)}")

    # GET method - just show the form
    return render_template('sicklecell.html', prediction=None)

# Diabetes prediction

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes_form():
    if request.method == 'POST':
        try:
            model = diabetes_model['model'] if isinstance(diabetes_model, dict) else diabetes_model
            input_values = [
                request.form['age'],
                request.form['gender'],
                request.form['Polyuria'],
                request.form['Polydipsia'],
                request.form['sudden weight loss'],
                request.form['weakness'],
                request.form['Polyphagia'],
                request.form['Genital thrush'],
                request.form['visual blurring'],
                request.form['Itching'],
                request.form['Irritability'],
                request.form['delayed healing'],
                request.form['partial paresis'],
                request.form['muscle stiffness'],
                request.form['Alopecia'],
                request.form['Obesity']
            ]

            input_array = np.array(input_values).reshape(1, -1)
            prediction = model.predict(input_array)

            if prediction[0] == 1:
                result = "You Have Diabetes"
            else:
                result = "You Don't Have. But Precaution is Better Than Cure :)"
            
            # Convert inputs to JSON
            feature_names = ['age', 'gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
                             'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching',
                             'Irritability', 'delayed healing', 'partial paresis', 'muscle stiffness',
                             'Alopecia', 'Obesity']
            input_dict = dict(zip(feature_names, input_values))
            params_json = json.dumps(input_dict)

            # Save to DB
            cursor = mysql.connection.cursor()
            user_id = session.get('user_id')
            cursor.execute("""
                INSERT INTO parameters_input (user_id, model_name, prediction, parameters_json, uploaded_at)
                VALUES (%s, %s, %s, %s, %s)""",
                (user_id, 'diabetes', result, params_json, datetime.now()))
            mysql.connection.commit()
            cursor.close()

            return render_template('diabetes.html', prediction=result)

        except Exception as e:
            return render_template('diabetes.html', prediction=f"Error: {str(e)}")

    # GET method - just show the form
    return render_template('diabetes.html', prediction=None)

# chatbot

api_key = os.getenv("GROK_API_KEY")
client = Groq(api_key=api_key)

# A function to check if the message contains health-related keywords
def is_health_related(message: str) -> bool:
    health_keywords = ["health", "disease", "medical", "condition", "illness", "symptoms", "symptom", "treatment", "diagnosis", "doctor", "medicine", "diabetes", "cancer", "obesity", "diet", 
"wellness", "prevention", "fitness", "lifestyle", "exercise", "nutrition", "well-being", "mental health", "healthcare", "heart disease", "stroke", "hypertension", "asthma", 
"arthritis", "infection", "influenza", "COVID-19", "flu", "autoimmune", "anemia", "cholesterol", "acne", "migraine", "allergy", "cough", "fever", "pneumonia", "surgery", 
"medication", "vaccine", "immunization", "therapy", "chemotherapy", "radiotherapy", "physiotherapy", "treatment options", "rehabilitation", "blood test", "check-up", "prescription", 
"diagnostic test", "blood pressure", "blood sugar", "heart rate", "cholesterol levels", "BMI", "calories", "vitamins", "minerals", "protein", "weight loss", "headache", "nausea", 
"dizziness", "fatigue", "pain", "coughing", "shortness of breath", "swelling", "rash", "vomiting", "chills", "cold sores", "depression", "anxiety", "stress", "insomnia", 
"bipolar disorder", "schizophrenia", "PTSD", "addiction", "mood swings", "therapy", "counseling", "vaccination", "screening", "health checkup", "health tips", "immunity", 
"health risks", "self-care", "healthy habits"]
    message = message.lower()  # Convert to lowercase to make the check case-insensitive
    return any(keyword in message for keyword in health_keywords)

def greet(message: str) -> bool:
    greet_keywords =["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening", "howdy", "welcome", "hi there", 
                     "hey there", "what's up", "salutations", "how's it going", "what is your name", "who are you"]

    message = message.lower()  # Convert to lowercase to make the check case-insensitive
    return any(keyword in message for keyword in greet_keywords)

# A function to check if the message contains usage or error-related keywords
def is_usage_or_error_related(message: str) -> bool:
    usage_keywords = ["how to use", "instructions", "guide", "help", "error", "issue", "problem", "bug", "report"]
    message = message.lower()  # Convert to lowercase to make the check case-insensitive
    return any(keyword in message for keyword in usage_keywords)

def is_goodbye(message: str) -> bool:
    goodbye_keywords = ["bye", "goodbye", "thank you", "thanks", "see you", "see ya", "take care", "farewell", "later", "talk to you later"]
    message = message.lower()
    return any(keyword in message for keyword in goodbye_keywords)

def creator(message: str) -> bool:
    creator_words = ['made you', 'created you', 'developed you', 'designed you', 'creator']
    message = message.lower()
    return any(keyword in message for keyword in creator_words)

def generate_bot_response(user_message: str) -> str:
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[ 
            {"role": "system", "content": "You are a friendly healthcare-assistant bot."},
            {"role": "user", "content": user_message},
        ],
        max_tokens=150,
        temperature=0.7,
    )
    bot_reply = completion.choices[0].message.content.strip()
    return clean_markdown(bot_reply)  # <<< clean here before returning

def clean_markdown(text: str) -> str:
    # Remove **bold**, *italic*, __underline__, etc.
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # remove bold **
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # remove italic *
    text = re.sub(r'__(.*?)__', r'\1', text)      # remove underline __
    text = re.sub(r'_([^_]*)_', r'\1', text)      # remove italic _
    return text

# Chat endpoint
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_msg = data.get("message", "").strip()

    if not user_msg:
        return jsonify({"response": "Please send a message."}), 400

    # Check if the message is health-related
    if is_health_related(user_msg):
        try:
            bot_reply = generate_bot_response(user_msg)
        except Exception as e:
            app.logger.error(f"Chat error: {e}")
            bot_reply = "Sorry, I'm having trouble right now. Please try again later."
        return jsonify({"response": bot_reply})

    # Check if the message is about usage or error-related queries
    if is_usage_or_error_related(user_msg):
        return jsonify({"response": "To use this webapp, "
        " - Visit Home"
        " - Click 'Start Diagnosing'"
        " - Select the disease that you want to predict"
        " - Click on 'Generate Prediction.' "
        "If you're encountering any issues, please report them here, and we'll get back to you."}), 200

    if greet(user_msg):
        return jsonify({"response": "Hello! I am HealthAI Bot to assist you with any health-related questions or queries. Feel free to ask about symptoms, diseases, treatments, and more. If you are facing any issues or need help with using the app, just let me know!"}), 200

    if is_goodbye(user_msg):
        return jsonify({"response": "Thank you for visiting! Stay healthy and take care. If you have more health-related questions later, feel free to return!"}), 200

    if creator(user_msg):
        return jsonify({"response": "Arya Singh and Deo Prakash are my creator. They designed me to help you. :) "})

    # If it's neither health nor usage-related, inform the user
    return jsonify({"response": "Sorry, I can only assist with health and medical questions. If you need help with the app, please ask 'how to use' or report an issue."}), 200

#Brain Tumor

@app.route('/brain_tumor', methods=['GET', 'POST'])
def brain_tumor_form():
    if not os.path.exists(brain_tumor_dir):
        os.makedirs(brain_tumor_dir)

    prediction = None
    cloud_url = None

    if request.method == 'POST':
        if 'my_image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['my_image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Generate unique filename with microseconds
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        new_filename = f"bt_pred_{timestamp}.jpg"
        public_id = f"bt_pred_{timestamp}"
        folder = "brain_tumor_predictions"

        # Save uploaded file locally
        img_path = os.path.join(brain_tumor_dir, new_filename)
        file.save(img_path)

        print("Brain tumor image saved locally at:", img_path)

        # Perform prediction
        prediction = predict_brain_tumor_image(brain_tumor_model, img_path, bt_class_names)

        # Upload to Cloudinary
        result = cloudinary.uploader.upload(img_path, public_id=public_id, folder=folder)
        cloud_url = result['secure_url']
        print("Uploaded to Cloudinary:", cloud_url)

        # Save metadata to MySQL
        cursor = mysql.connection.cursor()
        user_id = session.get('user_id')
        cursor.execute("""
            INSERT INTO uploads (user_id, filename, file_type, cloud_url, prediction, uploaded_at)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (user_id, file.filename, 'brain_tumor', cloud_url, prediction, datetime.now()))
        mysql.connection.commit()
        cursor.close()

        # Store results in session
        session['brain_prediction'] = prediction
        session['brain_img_url'] = cloud_url

        return redirect(url_for('brain_tumor_form'))

    # On GET
    prediction = session.pop('brain_prediction', None)
    cloud_url = session.pop('brain_img_url', None)

    return render_template('brain_tumor.html', prediction=prediction, img_url=cloud_url)



# Preprocess function
def predict_brain_tumor_image(model, image_path, bt_class_names):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Match model input size
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    prediction = brain_tumor_model(img)
    return bt_class_names[np.argmax(prediction)]

@app.route('/static/brain_tumor_dir/<filename>')
def serve_brain_tumor_image(filename):
    return send_from_directory(brain_tumor_dir, filename)


#Chronic Kidney Disease Risk Prediction

@app.route("/ckd", methods=["GET", "POST"])
def ckd_form():
    prediction = None
    result = None  # Initialize result to avoid UnboundLocalError

    if request.method == "POST":
        try:
            input_values = []
            input_dict = {}
            for feature in ckd_imp_features:
                value = request.form.get(feature)
                if feature == 'htn':
                    processed_value = 1 if value.lower() in ['yes', '1', 'true'] else 0
                else:
                    processed_value = float(value)
                input_values.append(processed_value)
                input_dict[feature] = value  # Save original user input (not the processed one)

            prediction_numeric = ckd_model.predict([input_values])[0]
            prediction = ckd_label_encoder.inverse_transform([prediction_numeric])[0]
            if prediction == 'ckd':
                result = 'Risk of Chronic Kidney Disease'
            else:
                result = 'You are safe as for now.'

            # Convert input_dict to JSON
            params_json = json.dumps(input_dict)

            # Save to DB
            cursor = mysql.connection.cursor()
            user_id = session.get('user_id')
            cursor.execute("""
                INSERT INTO parameters_input (user_id, model_name, prediction, parameters_json, uploaded_at)
                VALUES (%s, %s, %s, %s, %s)
            """, (user_id, 'ckd', result, params_json, datetime.now()))
            mysql.connection.commit()
            cursor.close()
        
        except Exception as e:
            result = f"Error in input: {e}"  # Fix: assign to result instead of prediction

    return render_template("ckd.html", features=ckd_imp_features, prediction=result)

# 41 Diseases
@app.route('/common_disease', methods = ["GET", "POST"])
def common_disease_form():
    fields = [
        "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering", "chills", "joint_pain",
    "stomach_pain", "acidity", "ulcers_on_tongue", "muscle_wasting", "vomiting", "burning_micturition",
    "spotting_urination", "fatigue", "weight_gain", "anxiety", "cold_hands_and_feets", "mood_swings",
    "weight_loss", "restlessness", "lethargy", "patches_in_throat", "irregular_sugar_level", "cough",
    "high_fever", "sunken_eyes", "breathlessness", "sweating", "dehydration", "indigestion", "headache",
    "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "pain_behind_the_eyes", "back_pain",
    "constipation", "abdominal_pain", "diarrhoea", "mild_fever", "yellow_urine", "yellowing_of_eyes",
    "acute_liver_failure", "fluid_overload", "swelling_of_stomach", "swelled_lymph_nodes", "malaise",
    "blurred_and_distorted_vision", "phlegm", "throat_irritation", "redness_of_eyes", "sinus_pressure",
    "runny_nose", "congestion", "chest_pain", "weakness_in_limbs", "fast_heart_rate",
    "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool", "irritation_in_anus", "neck_pain",
    "dizziness", "cramps", "bruising", "obesity", "swollen_legs", "swollen_blood_vessels", "puffy_face_and_eyes",
    "enlarged_thyroid", "brittle_nails", "swollen_extremeties", "excessive_hunger", "extra_marital_contacts",
    "drying_and_tingling_lips", "slurred_speech", "knee_pain", "hip_joint_pain", "muscle_weakness", "stiff_neck",
    "swelling_joints", "movement_stiffness", "spinning_movements", "loss_of_balance", "unsteadiness",
    "weakness_of_one_body_side", "loss_of_smell", "bladder_discomfort", "foul_smell_of urine",
    "continuous_feel_of_urine", "passage_of_gases", "internal_itching", "toxic_look_(typhos)", "depression",
    "irritability", "muscle_pain", "altered_sensorium", "red_spots_over_body", "belly_pain",
    "abnormal_menstruation", "dischromic_patches", "watering_from_eyes", "increased_appetite", "polyuria",
    "family_history", "mucoid_sputum", "rusty_sputum", "lack_of_concentration", "visual_disturbances",
    "receiving_blood_transfusion", "receiving_unsterile_injections", "coma", "stomach_bleeding",
    "distention_of_abdomen", "history_of_alcohol_consumption", "fluid_overload", "blood_in_sputum",
    "prominent_veins_on_calf", "palpitations", "painful_walking", "pus_filled_pimples", "blackheads",
    "scurring", "skin_peeling", "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails", "blister",
    "red_sore_around_nose", "yellow_crust_ooze"
    ]
    
    prediction = None
    if request.method == "POST":
        try:
            # Store selected symptoms only
            input_dict = {}
            input_values = []

            for symptom in fields:
                value = 1 if request.form.get(symptom) else 0
                input_dict[symptom] = value
                input_values.append(value)

            df_input = pd.DataFrame([input_values], columns=fields)

            # Predict
            probs = common_disease_model.predict(df_input)
            pred_idx = np.argmax(probs, axis=1)
            pred_label = common_disease_label_encoder.inverse_transform(pred_idx)
            prediction = pred_label[0]

            # SQL Save
            selected_symptoms = {k: v for k, v in input_dict.items() if v == 1}
            params_json = json.dumps(selected_symptoms)

            cursor = mysql.connection.cursor()
            user_id = session.get('user_id')
            cursor.execute("""
                INSERT INTO parameters_input (user_id, model_name, prediction, parameters_json, uploaded_at)
                VALUES (%s, %s, %s, %s, %s)
            """, (user_id, 'common_disease', prediction, params_json, datetime.now()))
            mysql.connection.commit()
            cursor.close()
            return render_template('common_disease.html', prediction=prediction) 

        except Exception as e:
            return render_template('common_disease.html', prediction=f"Error during prediction: {e}") 
    return render_template('common_disease.html', prediction=None)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(port=port, host="0.0.0.0", debug=True)

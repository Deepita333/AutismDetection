# AutismDetection
##🧠 Autism Detection Web App
This is a Flask-based web application that screens individuals for autism spectrum disorder (ASD) using the Q-CHAT-10 questionnaire. The application uses a machine learning model trained on real data to predict the likelihood of autism based on user responses.

AUTISM/
│
├── app.py                           # Main Flask application
├── autism_detection_qchat_model.pkl# Trained ML model
├── scaler_qchat.pkl                # Scaler used during model training
├── Autism.csv                      # Dataset used for training
├── detailed_assessment_model.pkl   # (Optional) For extended assessments
├── templates/                      # HTML templates for the web app
│   ├── home.html
│   ├── qchat.html
│   └── result.html
├── static/                         # Static files (CSS/JS)
├── uploads/                        # Directory for any uploads (if used)
└── app.log                         # Log file (for debugging/errors)

##🚀 Features
📋 Patient Form: Capture patient demographics and background information.

🤖 Q-CHAT-10 Screening: Users complete the 10-question autism screening test.

🧠 ML-Based Prediction: Responses are processed and classified using a pre-trained machine learning model.

📊 Results Page: Displays the user's autism likelihood score or classification.

🗃️ Modular Codebase: Easy to expand with video/image analysis or detailed assessments.

##🧪 Tech Stack
Backend: Python, Flask
ML Libraries: scikit-learn, joblib, numpy
Frontend: HTML, CSS (via templates)
Deployment: Localhost (development mode)

##🛠️ How to Run the Project

Clone the repository
git clone https://github.com/yourusername/autism-screening-app.git
cd autism-screening-app

Install dependencies
pip install flask scikit-learn numpy joblib

Run the application
python app.py

Open your browser and go to:
http://localhost:5000/

##📌 Disclaimer
This tool is a screening aid only. It is not a medical diagnostic tool. Please consult a healthcare professional for a full diagnosis.

##🙌 Author
Developed by Deepita Pradhan

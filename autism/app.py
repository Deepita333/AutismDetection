from flask import Flask, render_template, request, redirect, url_for
import joblib  # Load the trained model
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("autism_detection_qchat_model.pkl")
scaler = joblib.load("scaler_qchat.pkl")





# Home Page (Patient Details Form)
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get form data
        patient_name = request.form["name"]
        age = request.form["age"]
        gender = request.form["gender"]
        ethnicity = request.form["ethnicity"]
        jaundice = request.form["jaundice"]
        family_asd = request.form["family_asd"]
        completed_by = request.form["completed_by"]

        # Redirect to Q-CHAT test, passing these values
        return redirect(url_for("qchat", age=age, gender=gender, ethnicity=ethnicity,
                                jaundice=jaundice, family_asd=family_asd, completed_by=completed_by))
    return render_template("home.html")

# Q-CHAT Screening Page
@app.route("/qchat", methods=["GET"])
def qchat():
    # Pass the user data from query parameters to the template
    return render_template("qchat.html", 
                           age=request.args.get("age"),
                           gender=request.args.get("gender"),
                           ethnicity=request.args.get("ethnicity"),
                           jaundice=request.args.get("jaundice"),
                           family_asd=request.args.get("family_asd"),
                           completed_by=request.args.get("completed_by"))

@app.route("/result")
def result():
    score = request.args.get("score")
    return render_template("result.html", message=f"Total Score: {score}.")



# Detailed Assessment Page

if __name__ == "__main__":
    app.run(debug=True)

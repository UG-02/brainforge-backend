from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv
from ml_model import predict_performance
import os

load_dotenv()

app = Flask(__name__)
CORS(app)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.3-70b-versatile"

def ask_ai(prompt):
    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role":"system","content":"You are a helpful student AI assistant."},
                {"role":"user","content":prompt}
            ],
            temperature=0.7
        )
        return res.choices[0].message.content

    except Exception as e:
        return "AI Error: " + str(e)

@app.route("/")
def home():
    return jsonify({"status":"ok"})

@app.route("/notes", methods=["POST"])
def notes():
    data = request.get_json()
    topic = data.get("topic","")
    return jsonify({"result": ask_ai("Create exam notes on " + topic)})

@app.route("/planner", methods=["POST"])
def planner():
    data = request.get_json()
    goal = data.get("goal","")
    return jsonify({"result": ask_ai("Create study plan for " + goal)})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    hours = data.get("hours")
    difficulty = data.get("difficulty")
    consistency = data.get("consistency")
    
    score, advice = predict_performance(hours, difficulty, consistency)

    return jsonify({
    "predicted_score": score,
    "advice": advice
    })

    

@app.route("/quiz", methods=["POST"])
def quiz():
    data = request.get_json()
    topic = data.get("topic","")
    return jsonify({"result": ask_ai("Create MCQ quiz on " + topic)})

@app.route("/roadmap", methods=["POST"])
def roadmap():
    data = request.get_json()
    goal = data.get("goal","")
    return jsonify({"result": ask_ai("Create roadmap for " + goal)})

@app.route("/career", methods=["POST"])
def career():
    data = request.get_json()
    goal = data.get("goal","")
    return jsonify({"result": ask_ai("Career plan for " + goal)})

@app.route("/motivate", methods=["POST"])
def motivate():
    return jsonify({"result": ask_ai("Give motivational quote for students")})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    msg = data.get("message","")
    return jsonify({"result": ask_ai(msg)})

@app.route("/test")
def test():
    score, advice = predict_performance(4, 2, 3)
    return jsonify({
    "predicted_score": score,
    "advice": advice
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

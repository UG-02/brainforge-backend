import numpy as np
from sklearn.linear_model import LinearRegression

# Training data (simple fake dataset)
X = np.array([
    [1, 1, 1],
    [2, 1, 2],
    [3, 2, 2],
    [4, 2, 3],
    [5, 3, 3]
])

y = np.array([40, 50, 60, 75, 90])

model = LinearRegression()
model.fit(X, y)

def predict_performance(hours, difficulty, consistency):
    data = np.array([[hours, difficulty, consistency]])
    prediction = model.predict(data)[0]
    
    prediction = max(0, min(100, prediction))
    score = round(prediction, 2)

    if score < 40:
        advice = "Increase study hours and consistency"
    elif score < 70:
        advice = "Maintain consistency and revise weak areas"
    else:
        advice = "Excellent! Keep doing what you're doing"

    return score, advice

from flask import Flask, request, jsonify
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# ================= DATA =================
DATA = {
    "easy": [
        ("Government launches education policy", 1),
        ("Scientists discover water on Mars", 1),
        ("Aliens landed in India", 0),
        ("Miracle cure found for all diseases", 0)
    ],
    "medium": [
        ("WHO releases global health report", 1),
        ("Secret lab creates immortality drug", 0),
        ("Economic growth shows mixed trends", 1),
        ("Hidden group controls world politics", 0)
    ],
    "hard": [
        ("AI impacts global employment trends", 1),
        ("Climate report highlights risks", 1),
        ("Unverified cure spreads online rapidly", 0),
        ("Conspiracy group manipulates economy secretly", 0)
    ]
}

# ================= TRAIN MODEL =================
texts, labels = [], []
for lvl in DATA:
    for t, l in DATA[lvl]:
        texts.append(t)
        labels.append(l)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression(max_iter=500)
model.fit(X, labels)

# ================= PREDICT =================
def predict(text):
    x = vectorizer.transform([text])
    prob = model.predict_proba(x)[0]
    label = int(np.argmax(prob))
    confidence = float(np.max(prob))
    return label, confidence, prob

# ================= ENV =================
class OpenEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.level = 0
        self.i = 0
        return self.state()

    def state(self):
        lvl = list(DATA.keys())[self.level]
        return DATA[lvl][self.i % len(DATA[lvl])][0]

    def step(self, action, confidence):
        lvl = list(DATA.keys())[self.level]
        true = DATA[lvl][self.i % len(DATA[lvl])][1]

        reward = 1.0 if action == true else -1.0
        reward += (confidence - 0.5) * 0.4

        self.i += 1
        if self.i % 4 == 0 and self.level < 2:
            self.level += 1

        done = self.i >= 12

        return {
            "next_state": "" if done else self.state(),
            "reward": reward,
            "done": done
        }

env = OpenEnv()

# ================= API =================
@app.route("/reset", methods=["POST"])
def reset():
    state = env.reset()
    return jsonify({"state": state})

@app.route("/step", methods=["POST"])
def step():
    data = request.json
    action = data["action"]
    confidence = data.get("confidence", 0.5)

    result = env.step(action, confidence)
    return jsonify(result)

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.json
    text = data["text"]

    label, confidence, prob = predict(text)
    return jsonify({
        "label": label,
        "confidence": confidence
    })

# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)

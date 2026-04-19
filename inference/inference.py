from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return "OK"

@app.route("/reset", methods=["POST"])
def reset():
    return jsonify({"status": "reset successful"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    capital = data.get("capital", 1000)

    allocation = {
        "stocks": capital * 0.5,
        "crypto": capital * 0.3,
        "cash": capital * 0.2
    }

    return jsonify({"allocation": allocation})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import openai  # or any other model you are using

app = Flask(__name__)
CORS(app)  # allows frontend (HTML/JS) to access backend


openai.api_key = "YOUR_API_KEY"

@app.route('/')
def home():
    # If you have an HTML frontend (index.html)
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message')

        # Example using OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_message}]
        )

        bot_reply = response['choices'][0]['message']['content']
        return jsonify({"reply": bot_reply})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)


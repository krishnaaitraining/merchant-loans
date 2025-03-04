from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)
from dotenv import load_dotenv
import os
# Initialize Flask app
app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

from openai import OpenAI

client = OpenAI(api_key=api_key)

@app.route('/generate_text', methods=['POST'])
def predict():
    logging.info("Request received")
    data = request.get_json()
    input_message = data['in_message']
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are an AI assistant."},
                {"role": "user", "content": input_message}
        ]
    )

    output = completion.choices[0].message
    logging.info(output)
    logging.info(output.content)
    return jsonify({'Generated Response': output.content})


@app.route('/chatresponse', methods=['POST'])
def chatresponse():
    logging.info("Request received")
    input_message = request.form["in_message"]
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are an AI assistant. Generate response formatted using HTML."},
                {"role": "user", "content": input_message}
        ]
    )

    output = completion.choices[0].message
    logging.info(output)
    logging.info(output.content)
    response_text = output.content
    return render_template("inputtest.html", **locals())



@app.route('/', methods=['GET'])
def main():

    return render_template("inputtest.html")

if __name__ == '__main__':
    app.run(debug=True)
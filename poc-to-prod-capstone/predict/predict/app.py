# predict/app.py
from flask import Flask, request, render_template_string
from predict.predict.run import TextPredictionModel

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Tags Prediction</title>
    <style>
        /* Add your custom CSS styles here */
        body {
            font-family: Arial, sans-serif;
        }
        h2 {
            color: #007BFF;
        }
        form {
            margin-top: 20px;
        }
        textarea {
            width: 100%;
            margin-bottom: 10px;
        }
        button {
            background-color: #28A745;
            color: white;
            padding: 10px;
            border: none;
            cursor: pointer;
        }
        div.predictions {
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <h2>Tag Prediction</h2>
    <form action="/" method="post">
        <textarea name="text" rows="4" placeholder="Type your StackOverflow title here..."></textarea>
        <button type="submit">Click here to predict</button>
    </form>
    {% if predictions %}
    <div class="predictions">
        <strong>Predictions:</strong> {{ predictions }}
    </div>
    {% endif %}
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = None
    if request.method == 'POST':
        text_list = [request.form['text']]
        model = TextPredictionModel.from_artefacts('C:\\POC\\poc-to-prod-capstone\\train\\data\\artefacts\\2024-01-09-21-10-14')
        predictions = model.predict(text_list)
    return render_template_string(HTML_TEMPLATE, predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)

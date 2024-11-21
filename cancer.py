from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'values')
)


# Load the model
with open('C:/Users/USER/SVM_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    description = None
    if request.method == 'POST':
        features = [float(request.form[f'feature_{i}']) for i in range(1, 31)]  # Now 30 features
        pred_value = model.predict([np.array(features)])[0]
         # Add explanation for prediction
        if pred_value == 1:
            prediction = "Malignant"
            description = "The model predicts a cancerous cell (malignant tumor). This may require further medical evaluation."
        else:
            prediction = "Benign"
            description = "The model predicts a non-cancerous cell (benign tumor). However, you should consult a doctor for confirmation."
    
    return render_template('view.html', prediction=prediction, description=description)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(data['features'])])
    return jsonify(prediction[0])

if __name__ == '__main__':
    app.run(port=5000, debug=True)

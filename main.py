from flask import Flask, request, jsonify
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import numpy as np

# Charger le dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Pipeline avec normalisation et Logistic Regression
model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=200, random_state=42))
])

# Entraîner le modèle
model.fit(X_train, y_train)

# Évaluer le modèle
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {accuracy:.4f}")

# Dictionnaire pour mapper les labels aux noms des espèces
species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}

# API Flask
app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Récupérer les paramètres de la requête
        sepal_length = request.args.get('sepal_length', type=float)
        sepal_width = request.args.get('sepal_width', type=float)
        petal_length = request.args.get('petal_length', type=float)
        petal_width = request.args.get('petal_width', type=float)
        
        if None in [sepal_length, sepal_width, petal_length, petal_width]:
            return jsonify({'error': 'Tous les paramètres (sepal_length, sepal_width, petal_length, petal_width) sont requis'}), 400
        
        # Construire les caractéristiques en tableau NumPy
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Faire la prédiction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = float(max(probabilities))
        
        # Construire la réponse
        response = {
            'confidence': confidence,
            'model_type': 'LogisticRegression',
            'parameters': {
                'sepal_length': sepal_length,
                'sepal_width': sepal_width,
                'petal_length': petal_length,
                'petal_width': petal_width
            },
            'prediction': int(prediction),
            'species': species_map[int(prediction)]
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

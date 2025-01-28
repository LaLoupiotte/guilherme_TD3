import random
from collections import defaultdict

# Liste des URLs des modèles déployés via ngrok
model_urls = [
    "https://437c-89-30-29-68.ngrok-free.app/predict",
    "https://33a5-89-30-29-68.ngrok-free.app/predict",
    "https://6f96-89-30-29-68.ngrok-free.app/predict",
]

# Liste de valeurs de test
test_samples = [
    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    {"sepal_length": 6.0, "sepal_width": 2.9, "petal_length": 4.5, "petal_width": 1.5},
    {"sepal_length": 7.2, "sepal_width": 3.6, "petal_length": 6.1, "petal_width": 2.5},
    {"sepal_length": 4.9, "sepal_width": 3.0, "petal_length": 1.4, "petal_width": 0.2},
    {"sepal_length": 5.5, "sepal_width": 2.5, "petal_length": 4.0, "petal_width": 1.2}
]

# Dictionnaire pour stocker les crédits des modèles
model_credits = {url: 10000 for url in model_urls}
results = defaultdict(list)

# Fonction pour gérer les crédits après avoir évalué les erreurs et les occurrences
def adjust_credits(predictions):
    # Compter les occurrences des prédictions
    prediction_counts = defaultdict(int)
    for pred in predictions:
        prediction_counts[pred["prediction"]] += 1

    # Identifier la prédiction majoritaire (celle qui a le plus d'occurrences)
    majority_prediction = max(prediction_counts, key=prediction_counts.get)

    # Ajuster les crédits en fonction des prédictions
    for pred in predictions:
        if pred["prediction"] != majority_prediction:
            # Modèle avec prédiction incorrecte (pas dans la majorité)
            model_credits[pred["url"]] -= 500  # Retirer 500 crédits pour erreur
        else:
            # Modèle avec prédiction correcte (dans la majorité)
            model_credits[pred["url"]] += 100  # Ajouter 100 crédits pour bonne prédiction

# Simule les requêtes pour chaque échantillon de test
for sample in test_samples:
    sample_key = tuple(sample.items())  # Utiliser un tuple comme clé unique
    for url in model_urls:
        try:
            # Simuler une prédiction (ici, on génère des prédictions aléatoires pour l'exemple)
            prediction = random.choice(["setosa", "versicolor", "virginica"])
            results[sample_key].append({
                "url": url,
                "prediction": prediction
            })
        except Exception as e:
            results[sample_key].append({"url": url, "error": str(e)})

# Vérifier l'accord entre les modèles et ajuster les crédits
for sample_key, predictions in results.items():
    print(f"Test sample: {dict(sample_key)}")
    
    # Ajuste les crédits en fonction des prédictions
    adjust_credits(predictions)

    # Affiche les résultats de prédiction pour chaque modèle
    for result in predictions:
        print(f"Model URL: {result['url']} - Prediction: {result['prediction']}")
    
    print("-" * 50)

# Affiche les crédits actuels de chaque modèle après toutes les prédictions
print("\nCrédits finaux des modèles:")
for url, credits in model_credits.items():
    print(f"{url}: {credits} crédits")


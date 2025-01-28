import requests
import time
from collections import defaultdict

# Liste des URLs des modèles déployés via ngrok
model_urls = [
    "https://437c-89-30-29-68.ngrok-free.app/predict",
    "https://33a5-89-30-29-68.ngrok-free.app/predict",
]
# Liste de valeurs de test
test_samples = [
    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    {"sepal_length": 6.0, "sepal_width": 2.9, "petal_length": 4.5, "petal_width": 1.5},
    {"sepal_length": 7.2, "sepal_width": 3.6, "petal_length": 6.1, "petal_width": 2.5}
]

results = defaultdict(list)

for sample in test_samples:
    sample_key = tuple(sample.items())  # Utiliser un tuple comme clé unique
    for url in model_urls:
        try:
            start_time = time.time()
            response = requests.get(url, params=sample)
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                results[sample_key].append({
                    "url": url,
                    "prediction": data.get("species", "Unknown"),
                    "confidence": data.get("confidence", 0.0),
                    "response_time": elapsed_time
                })
            else:
                results[sample_key].append({"url": url, "error": f"Erreur {response.status_code}"})
        except Exception as e:
            results[sample_key].append({"url": url, "error": str(e)})

# Vérifier l'accord entre les modèles
for sample_key, predictions in results.items():
    print(f"Test sample: {dict(sample_key)}")
    prediction_set = set(pred["prediction"] for pred in predictions if "prediction" in pred)
    if len(prediction_set) == 1:
        print("✅ Tous les modèles sont d'accord: ", prediction_set.pop())
    else:
        print("❌ Désaccord entre les modèles: ", prediction_set)
    for result in predictions:
        print(result)
    print("-" * 50)


# AI- Powered Disease Predictor Web-App

```
A web-based disease prediction tool built with Python, Flask, and scikit-learn. 
Users input their symptoms through a smart autocomplete interface and receive 
an AI-generated prediction with confidence score and disease description. 

The model is trained on a dataset of 41 diseases and ~130 symptoms using 
Random Forest, Decision Tree, and Naive Bayes classifiers. Predictions are 
refined using clinical post-filters based on onset, episode history, and 
neurological deficit.

Built with: Python · Flask · scikit-learn · HTML · CSS · JavaScript
```

## Running locally
```bash
pip install -r requirements.txt
py app.py
```
Then open http://localhost:5000

## API Endpoints
- GET  /symptoms  → returns list of all symptoms (for autocomplete)
- POST /predict   → streams prediction response

## POST /predict body example
```json
{
  "symptoms": ["itching", "skin rash", "nodal skin eruptions"],
  "onset": "gradual",
  "episode_history": "no",
  "neurological_deficit": "none"
}
```

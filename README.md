# Disease Predictor — Project Structure

```
disease_predictor/
│
├── app.py                  ← Flask backend (main file)
├── requirements.txt        ← Python dependencies
├── Procfile                ← For Render deployment
│
├── model/
│   ├── model.pkl           ← Trained ML model (copy from Colab)
│   ├── symptom_columns.json← Symptom column order (copy from Colab)
│   └── filter_rules.pkl    ← Post-filter rules (copy from Colab)
│
├── templates/
│   └── index.html          ← Frontend (Day 5)
│
└── static/
    ├── css/
    │   └── style.css       ← Styles (Day 5)
    └── js/
        └── app.js          ← Frontend logic (Day 5)
```

## Running locally
```bash
pip install -r requirements.txt
python app.py
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

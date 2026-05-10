import os
import re
import json
import time
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, Response, stream_with_context, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model & supporting files
BASE = os.path.dirname(__file__)

# Neural Network
import tensorflow as tf
nn_model = tf.keras.models.load_model(os.path.join(BASE, 'model', 'nn_model.keras'))

# Random Forest
rf_model = joblib.load(os.path.join(BASE, 'model', 'rf_model.pkl'))

# Label encoder & symptom columns
le           = joblib.load(os.path.join(BASE, 'model', 'label_encoder.pkl'))
symptom_cols = json.load(open(os.path.join(BASE, 'model', 'symptom_columns.json')))

# Disease info (severity + specialist)
disease_info = json.load(open(os.path.join(BASE, 'model', 'disease_info.json')))

# Load descriptions from CSV 
desc_df      = pd.read_csv(os.path.join(BASE, 'model', 'symptom_Description.csv'))
DISEASE_DESC = dict(zip(desc_df['Disease'].str.strip(), desc_df['Description'].str.strip()))
DEFAULT_DESC = "A medical condition identified based on your reported symptoms."

# NLP Synonym map
SYNONYM_MAP = {
    'dizzy':'dizziness','dizziness':'dizziness','spinning':'dizziness','vertigo':'dizziness',
    'lightheaded':'dizziness','light headed':'dizziness',
    'tired':'fatigue','tiredness':'fatigue','fatigue':'fatigue','exhausted':'fatigue',
    'exhaustion':'fatigue','no energy':'fatigue','weakness':'fatigue','weak':'fatigue',
    'lethargic':'lethargy','lethargy':'lethargy','sluggish':'lethargy',
    'headache':'headache','head ache':'headache','head pain':'headache',
    'migraine':'headache','head hurts':'headache','head hurting':'headache',
    'frequent urine':'polyuria','frequent urination':'polyuria','polyuria':'polyuria',
    'urinating frequently':'polyuria','urinating a lot':'polyuria','peeing a lot':'polyuria',
    'peeing frequently':'polyuria','increased urination':'polyuria','more urine':'polyuria',
    'high fever':'high fever','fever':'high fever','temperature':'high fever',
    'high temperature':'high fever','running a fever':'high fever',
    'mild fever':'mild fever','low grade fever':'mild fever','slight fever':'mild fever',
    'nausea':'nausea','nauseous':'nausea','feel like vomiting':'nausea','queasy':'nausea',
    'vomiting':'vomiting','vomit':'vomiting','throwing up':'vomiting','puking':'vomiting',
    'chills':'chills','shivering':'shivering','shivers':'shivering','trembling':'shivering',
    'feeling cold':'chills','cold feeling':'chills',
    'sweating':'sweating','sweat':'sweating','night sweats':'sweating',
    'itching':'itching','itchy':'itching','itchiness':'itching','scratching':'itching',
    'internal itching':'internal itching',
    'skin rash':'skin rash','rash':'skin rash','rashes':'skin rash','red skin':'skin rash',
    'stomach pain':'stomach pain','stomach ache':'stomach pain','tummy pain':'stomach pain',
    'abdominal pain':'abdominal pain','abdomen pain':'abdominal pain',
    'belly pain':'belly pain','belly ache':'belly pain',
    'chest pain':'chest pain','chest hurts':'chest pain','chest tightness':'chest pain',
    'back pain':'back pain','back ache':'back pain','backache':'back pain',
    'joint pain':'joint pain','joint ache':'joint pain','joints hurt':'joint pain',
    'muscle pain':'muscle pain','muscle ache':'muscle pain','body ache':'muscle pain','body pain':'muscle pain',
    'knee pain':'knee pain','neck pain':'neck pain','hip joint pain':'hip joint pain',
    'breathlessness':'breathlessness','shortness of breath':'breathlessness',
    'short of breath':'breathlessness','difficulty breathing':'breathlessness',
    'trouble breathing':'breathlessness','breathless':'breathlessness',
    'cough':'cough','coughing':'cough','dry cough':'cough','persistent cough':'cough',
    'phlegm':'phlegm','mucus':'phlegm','blood in sputum':'blood in sputum',
    'coughing blood':'blood in sputum',
    'diarrhea':'diarrhoea','diarrhoea':'diarrhoea','loose stool':'diarrhoea',
    'loose motions':'diarrhoea','watery stool':'diarrhoea',
    'constipation':'constipation','constipated':'constipation',
    'bloody stool':'bloody stool','blood in stool':'bloody stool',
    'burning micturition':'burning micturition','burning urine':'burning micturition',
    'pain urinating':'burning micturition','painful urination':'burning micturition',
    'bladder discomfort':'bladder discomfort','bladder pain':'bladder discomfort',
    'continuous feel of urine':'continuous feel of urine',
    'dark urine':'dark urine','yellow urine':'yellow urine',
    'foul smell urine':'foul smell of urine','smelly urine':'foul smell of urine',
    'yellowish skin':'yellowish skin','yellow skin':'yellowish skin','jaundice':'yellowish skin',
    'yellowing of eyes':'yellowing of eyes','yellow eyes':'yellowing of eyes',
    'skin peeling':'skin peeling','blackheads':'blackheads',
    'pimples':'pus filled pimples','pus filled pimples':'pus filled pimples','acne':'pus filled pimples',
    'nodal skin eruptions':'nodal skin eruptions','dischromic patches':'dischromic patches',
    'skin patches':'dischromic patches','red spots':'red spots over body',
    'blurred vision':'blurred and distorted vision','blurry vision':'blurred and distorted vision',
    'visual disturbances':'visual disturbances','redness of eyes':'redness of eyes',
    'red eyes':'redness of eyes','watery eyes':'watering from eyes',
    'pain behind eyes':'pain behind the eyes','puffy eyes':'puffy face and eyes',
    'sunken eyes':'sunken eyes',
    'loss of appetite':'loss of appetite','no appetite':'loss of appetite','not hungry':'loss of appetite',
    'increased appetite':'increased appetite','always hungry':'excessive hunger',
    'excessive hunger':'excessive hunger',
    'weight loss':'weight loss','losing weight':'weight loss',
    'weight gain':'weight gain','gaining weight':'weight gain','obesity':'obesity',
    'thirst':'dehydration','thirsty':'dehydration','very thirsty':'dehydration',
    'always thirsty':'dehydration','dehydration':'dehydration','dehydrated':'dehydration',
    'anxiety':'anxiety','anxious':'anxiety','nervous':'anxiety',
    'depression':'depression','depressed':'depression',
    'mood swings':'mood swings','irritability':'irritability','irritable':'irritability',
    'restlessness':'restlessness','restless':'restlessness',
    'lack of concentration':'lack of concentration','cant concentrate':'lack of concentration',
    'insomnia':'restlessness','cant sleep':'restlessness','trouble sleeping':'restlessness',
    'sore throat':'throat irritation','throat pain':'throat irritation','throat irritation':'throat irritation',
    'runny nose':'runny nose','nose running':'runny nose',
    'congestion':'congestion','stuffy nose':'congestion','blocked nose':'congestion',
    'sneezing':'continuous sneezing','continuous sneezing':'continuous sneezing',
    'sinus pressure':'sinus pressure','loss of smell':'loss of smell',
    'palpitations':'palpitations','heart pounding':'palpitations',
    'fast heartbeat':'fast heart rate','fast heart rate':'fast heart rate','racing heart':'fast heart rate',
    'swollen legs':'swollen legs','leg swelling':'swollen legs',
    'bruising':'bruising','bruise':'bruising',
    'stiff neck':'stiff neck','neck stiffness':'stiff neck',
    'movement stiffness':'movement stiffness','stiffness':'movement stiffness',
    'muscle weakness':'muscle weakness','weakness in limbs':'weakness in limbs',
    'weak limbs':'weakness in limbs','loss of balance':'loss of balance',
    'unsteadiness':'unsteadiness','unsteady':'unsteadiness',
    'cramps':'cramps','muscle cramps':'cramps','swelling joints':'swelling joints',
    'indigestion':'indigestion','acidity':'acidity','acid reflux':'acidity','heartburn':'acidity',
    'gas':'passage of gases','bloating':'distention of abdomen','bloated':'distention of abdomen',
    'swollen stomach':'swelling of stomach',
    'confusion':'altered sensorium','confused':'altered sensorium','disoriented':'altered sensorium',
    'slurred speech':'slurred speech','speech problems':'slurred speech',
    'enlarged thyroid':'enlarged thyroid','swollen lymph nodes':'swelled lymph nodes',
    'brittle nails':'brittle nails','irregular sugar level':'irregular sugar level',
    'blood sugar irregular':'irregular sugar level','alcohol':'history of alcohol consumption',
    'abnormal menstruation':'abnormal menstruation','irregular periods':'abnormal menstruation',
}

def extract_symptoms_from_text(text):
    text_lower = text.lower().strip()
    text_lower = re.sub(r'[^a-z\s]', ' ', text_lower)
    text_lower = re.sub(r'\s+', ' ', text_lower).strip()
    matched = set()
    for phrase in sorted(SYNONYM_MAP.keys(), key=len, reverse=True):
        if re.search(r'\b' + re.escape(phrase) + r'\b', text_lower):
            target = SYNONYM_MAP[phrase]
            if target in symptom_cols:
                matched.add(target)
    for symptom in symptom_cols:
        sym_lower = symptom.lower().strip()
        if symptom in matched:
            continue
        if re.search(r'\b' + re.escape(sym_lower) + r'\b', text_lower):
            matched.add(symptom)
    return sorted(matched)


# Routes 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    return jsonify(sorted(symptom_cols))

@app.route('/disease_info', methods=['GET'])
def get_disease_info():
    return jsonify(disease_info)

@app.route('/extract-text', methods=['POST'])
def extract_text():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Only PDF files supported'}), 400
    try:
        import pypdf
        reader = pypdf.PdfReader(file)
        text = ' '.join(page.extract_text() or '' for page in reader.pages)
        return jsonify({'text': text.strip()})
    except Exception as e:
        return jsonify({'error': f'PDF extraction failed: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    data      = request.get_json(force=True)
    free_text = data.get('text', '').strip()
    symptoms  = data.get('symptoms', [])

    if free_text:
        valid_symptoms = extract_symptoms_from_text(free_text)
    else:
        valid_symptoms = [s for s in symptoms if s in symptom_cols]

    if not valid_symptoms:
        return jsonify({'error': 'Could not extract any recognizable symptoms. Please be more specific, e.g. "I have headache, fever and nausea".'}), 400
    if len(valid_symptoms) < 2:
        return jsonify({'error': 'Please describe at least 2 symptoms for an accurate prediction.'}), 400

    # Build input vector
    vec = np.array([[1 if s in valid_symptoms else 0 for s in symptom_cols]])

    # NN (60%) + RF (40%) ensemble
    nn_proba = nn_model.predict(vec, verbose=0)[0]
    rf_proba = rf_model.predict_proba(vec)[0]
    ensemble = 0.6 * nn_proba + 0.4 * rf_proba

    top5_idx  = ensemble.argsort()[-5:][::-1]
    top5_dis  = [le.inverse_transform([i])[0] for i in top5_idx]
    top5_pro  = np.array([ensemble[i] for i in top5_idx])
    top5_norm = top5_pro / top5_pro.sum()
    ranked    = list(zip(top5_dis, top5_norm))

    top_disease, top_score = ranked[0]
    description = DISEASE_DESC.get(top_disease, DEFAULT_DESC)
    info        = disease_info.get(top_disease, {})
    severity    = info.get('severity',   'Unknown')
    specialist  = info.get('specialist', 'General Physician')

    lines = []
    lines.append(f"## Predicted Condition\n**{top_disease}**\n")
    lines.append(f"### Extracted Symptoms\n{', '.join(valid_symptoms)}\n")
    lines.append(f"### Confidence Score\n{top_score * 100:.1f}%\n")
    lines.append(f"### Severity\n{severity}\n")
    lines.append(f"### Recommended Specialist\n{specialist}\n")
    lines.append(f"### About This Condition\n{description}\n")
    lines.append("### Other Possibilities")
    for disease, score in ranked[1:4]:
        lines.append(f"- **{disease}** — {score*100:.1f}%")
    lines.append("")
    lines.append("---")
    lines.append("*⚠️ This tool is for educational purposes only. Always consult a qualified medical professional.*")

    full_text = "\n".join(lines)

    def generate():
        words = full_text.split(' ')
        for i, word in enumerate(words):
            chunk = word + (' ' if i < len(words) - 1 else '')
            yield f"data: {json.dumps(chunk)}\n\n"
            time.sleep(0.02)
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
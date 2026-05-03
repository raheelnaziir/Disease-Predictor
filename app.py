import os
import json
import time
import pickle
import joblib
import numpy as np
from flask import Flask, request, jsonify, Response, stream_with_context, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Load model & supporting files ──────────────────────────────────────────
BASE = os.path.dirname(__file__)

model          = joblib.load(os.path.join(BASE, 'model', 'model.pkl'))
symptom_cols   = json.load(open(os.path.join(BASE, 'model', 'symptom_columns.json')))
filter_rules   = pickle.load(open(os.path.join(BASE, 'model', 'filter_rules.pkl'), 'rb'))

ALL_SYMPTOMS   = sorted(symptom_cols)   # used for autocomplete


# ── Disease descriptions (shown in streaming response) ────────────────────
DISEASE_INFO = {
    "Fungal infection"        : "A fungal infection caused by dermatophytes affecting skin, nails, or scalp.",
    "Allergy"                 : "An immune system reaction to a foreign substance such as pollen, food, or insect stings.",
    "GERD"                    : "Gastroesophageal Reflux Disease — stomach acid frequently flows back into the esophagus.",
    "Chronic cholestasis"     : "A liver condition where bile flow is reduced or blocked, causing itching and jaundice.",
    "Drug Reaction"           : "An adverse reaction caused by medication, ranging from mild rash to severe anaphylaxis.",
    "Peptic ulcer disease"    : "Sores on the stomach lining or the upper part of the small intestine.",
    "AIDS"                    : "Advanced stage of HIV infection that severely damages the immune system.",
    "Diabetes"                : "A metabolic disease causing high blood sugar due to insulin deficiency or resistance.",
    "Gastroenteritis"         : "Inflammation of the stomach and intestines, typically causing vomiting and diarrhea.",
    "Bronchial Asthma"        : "A chronic condition where airways narrow and swell, producing extra mucus.",
    "Hypertension"            : "Persistently high blood pressure that can lead to heart disease and stroke.",
    "Migraine"                : "A neurological condition causing intense headaches, often with nausea and light sensitivity.",
    "Cervical spondylosis"    : "Age-related wear affecting spinal disks in the neck causing pain and stiffness.",
    "Paralysis (brain hemorrhage)": "Loss of muscle function caused by bleeding in the brain.",
    "Jaundice"                : "Yellowing of the skin and eyes caused by excess bilirubin in the blood.",
    "Malaria"                 : "A mosquito-borne infectious disease caused by Plasmodium parasites.",
    "Chicken pox"             : "A highly contagious viral infection causing an itchy blister-like rash.",
    "Dengue"                  : "A mosquito-borne viral infection causing high fever, rash, and muscle pain.",
    "Typhoid"                 : "A bacterial infection caused by Salmonella typhi spread through contaminated food or water.",
    "hepatitis A"             : "A liver infection caused by the Hepatitis A virus, spread through contaminated food or water.",
    "Hepatitis B"             : "A serious liver infection caused by the Hepatitis B virus, transmitted through bodily fluids.",
    "Hepatitis C"             : "A viral infection causing liver inflammation, sometimes leading to serious liver damage.",
    "Hepatitis D"             : "A liver disease caused by the Hepatitis D virus — only occurs with Hepatitis B.",
    "Hepatitis E"             : "A liver disease caused by the Hepatitis E virus, spread through contaminated water.",
    "Alcoholic hepatitis"     : "Liver inflammation caused by excessive alcohol consumption.",
    "Tuberculosis"            : "A serious bacterial infection mainly affecting the lungs, spread through the air.",
    "Common Cold"             : "A viral infection of the upper respiratory tract causing runny nose and sore throat.",
    "Pneumonia"               : "Infection inflaming air sacs in one or both lungs, which may fill with fluid.",
    "Dimorphic hemmorhoids(piles)": "Swollen veins in the rectum or anus causing pain, bleeding, and discomfort.",
    "Heart attack"            : "A blockage of blood flow to the heart muscle, a medical emergency.",
    "Varicose veins"          : "Enlarged, twisted veins usually appearing in the legs due to faulty valves.",
    "Hypothyroidism"          : "Underactive thyroid gland failing to produce enough thyroid hormone.",
    "Hyperthyroidism"         : "Overactive thyroid producing too much thyroxine, accelerating metabolism.",
    "Hypoglycemia"            : "Abnormally low blood sugar level causing shakiness, confusion, and dizziness.",
    "Osteoarthritis"          : "Degeneration of joint cartilage and bone causing pain and stiffness.",
    "Arthritis"               : "Inflammation of one or more joints causing pain and stiffness.",
    "(vertigo) Paroxysmal Positional Vertigo": "A disorder of the inner ear causing brief episodes of dizziness.",
    "Acne"                    : "A skin condition causing pimples, blackheads, and whiteheads on the face and body.",
    "Urinary tract infection" : "An infection in any part of the urinary system, most commonly the bladder.",
    "Psoriasis"               : "A skin disease causing red, itchy, scaly patches on the skin.",
    "Impetigo"                : "A highly contagious bacterial skin infection causing red sores that rupture.",
}

DEFAULT_INFO = "A medical condition identified based on your reported symptoms and clinical indicators."

RECOMMENDATIONS = {
    "sudden":     " Given the sudden onset, please seek medical attention promptly.",
    "gradual":    " With a gradual onset, schedule a consultation with your doctor soon.",
    "persistent": " Persistent neurological symptoms require urgent neurological evaluation.",
    "yes":        " Recurring episodes suggest a chronic condition — specialist follow-up is recommended.",
}


# ── Helper: post-filter ────────────────────────────────────────────────────
def apply_post_filter(top_diseases, top_probs, extra):
    scores = {d: p for d, p in zip(top_diseases, top_probs)}
    for field, value in extra.items():
        if not value or field not in filter_rules:
            continue
        rules = filter_rules[field].get(value.lower(), {})
        for disease in list(scores.keys()):
            d_lower = disease.lower()
            for kw in rules.get('boost', []):
                if kw in d_lower:
                    scores[disease] = min(1.0, scores[disease] + 0.15)
            for kw in rules.get('penalise', []):
                if kw in d_lower:
                    scores[disease] = max(0.0, scores[disease] - 0.20)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    """Return full symptom list for autocomplete."""
    return jsonify(ALL_SYMPTOMS)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON body:
    {
      "symptoms": ["itching", "skin rash"],
      "onset": "gradual",
      "episode_history": "yes",
      "neurological_deficit": "none"
    }
    Returns a streaming plain-text response (Server-Sent Events).
    """
    data   = request.get_json(force=True)
    symptoms = data.get('symptoms', [])
    extra    = {
        'onset'               : data.get('onset', ''),
        'episode_history'     : data.get('episode_history', ''),
        'neurological_deficit': data.get('neurological_deficit', '')
    }

    if not symptoms:
        return jsonify({'error': 'No symptoms provided'}), 400

    # Build input vector
    vec = [1 if s in symptoms else 0 for s in symptom_cols]
    input_df = __import__('pandas').DataFrame([vec], columns=symptom_cols)

    # Top-5 predictions
    proba    = model.predict_proba(input_df)[0]
    top5_idx = proba.argsort()[-5:][::-1]
    top5_dis = [model.classes_[i] for i in top5_idx]
    top5_pro = [float(proba[i])   for i in top5_idx]

    ranked   = apply_post_filter(top5_dis, top5_pro, extra)
    top_disease, top_score = ranked[0]
    description = DISEASE_INFO.get(top_disease, DEFAULT_INFO)

    # Build the full response text
    lines = []
    lines.append(f"## Predicted Condition\n**{top_disease}**\n")
    lines.append(f"### Confidence Score\n{top_score * 100:.1f}%\n")
    lines.append(f"### About This Condition\n{description}\n")

    lines.append("### Other Possibilities")
    for disease, score in ranked[1:4]:
        lines.append(f"- **{disease}** — {score*100:.1f}%")
    lines.append("")

    # Add contextual recommendation
    recs = []
    if extra['onset']:
        r = RECOMMENDATIONS.get(extra['onset'])
        if r: recs.append(r)
    if extra['neurological_deficit'] == 'persistent':
        recs.append(RECOMMENDATIONS['persistent'])
    if extra['episode_history'] == 'yes':
        recs.append(RECOMMENDATIONS['yes'])

    if recs:
        lines.append("### Recommendations")
        lines += recs
        lines.append("")

    lines.append("---")
    lines.append("*⚠️ This tool is for educational purposes only. Always consult a qualified medical professional.*")

    full_text = "\n".join(lines)

    # ── Stream word by word ───────────────────────────────────────────────
    def generate():
        words = full_text.split(' ')
        for i, word in enumerate(words):
            chunk = word + (' ' if i < len(words) - 1 else '')
            yield f"data: {json.dumps(chunk)}\n\n"
            time.sleep(0.04)   # adjust speed here (seconds per word)
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control' : 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


if __name__ == '__main__':
    app.run(debug=True, threaded=True)

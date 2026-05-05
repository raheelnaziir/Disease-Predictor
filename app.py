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

# Disease descriptions (shown in streaming response) 
DISEASE_INFO = {
    "Fungal infection"        : "In humans, fungal infections occur when an invading fungus takes over an area of the body and is too much for the immune system to handle. Fungi can live in the air, soil, water, and plants. There are also some fungi that live naturally in the human body. Like many microbes, there are helpful fungi and harmful fungi.",
    "Allergy"                 : "An allergy is an immune system response to a foreign substance that's not typically harmful to your body.They can include certain foods, pollen, or pet dander. Your immune system's job is to keep you healthy by fighting harmful pathogens.",
    "GERD"                    : "Gastroesophageal reflux disease, or GERD, is a digestive disorder that affects the lower esophageal sphincter (LES), the ring of muscle between the esophagus and stomach. Many people, including pregnant women, suffer from heartburn or acid indigestion caused by GERD.",
    "Chronic cholestasis"     : "Chronic cholestatic diseases, whether occurring in infancy, childhood or adulthood, are characterized by defective bile acid transport from the liver to the intestine, which is caused by primary damage to the biliary epithelium in most cases",
    "Drug Reaction"           : "An adverse drug reaction (ADR) is an injury caused by taking medication. ADRs may occur following a single dose or prolonged administration of a drug or result from the combination of two or more drugs.",
    "Peptic ulcer disease"    : "Peptic ulcer disease (PUD) is a break in the inner lining of the stomach, the first part of the small intestine, or sometimes the lower esophagus. An ulcer in the stomach is called a gastric ulcer, while one in the first part of the intestines is a duodenal ulcer.",
    "AIDS"                    : "Acquired immunodeficiency syndrome (AIDS) is a chronic, potentially life-threatening condition caused by the human immunodeficiency virus (HIV). By damaging your immune system, HIV interferes with your body's ability to fight infection and disease.",
    "Diabetes"                : "Diabetes is a disease that occurs when your blood glucose, also called blood sugar, is too high. Blood glucose is your main source of energy and comes from the food you eat. Insulin, a hormone made by the pancreas, helps glucose from food get into your cells to be used for energy.",
    "Gastroenteritis"         : "Gastroenteritis is an inflammation of the digestive tract, particularly the stomach, and large and small intestines. Viral and bacterial gastroenteritis are intestinal infections associated with symptoms of diarrhea , abdominal cramps, nausea , and vomiting .",
    "Bronchial Asthma"        : "Bronchial asthma is a medical condition which causes the airway path of the lungs to swell and narrow. Due to this swelling, the air path produces excess mucus making it hard to breathe, which results in coughing, short breath, and wheezing. The disease is chronic and interferes with daily working.",
    "Hypertension"            : "Hypertension (HTN or HT), also known as high blood pressure (HBP), is a long-term medical condition in which the blood pressure in the arteries is persistently elevated. High blood pressure typically does not cause symptoms.",
    "Migraine"                : "A migraine can cause severe throbbing pain or a pulsing sensation, usually on one side of the head. It's often accompanied by nausea, vomiting, and extreme sensitivity to light and sound. Migraine attacks can last for hours to days, and the pain can be so severe that it interferes with your daily activities.",
    "Cervical spondylosis"    : "Cervical spondylosis is a general term for age-related wear and tear affecting the spinal disks in your neck. As the disks dehydrate and shrink, signs of osteoarthritis develop, including bony projections along the edges of bones (bone spurs).",
    "Paralysis (brain hemorrhage)": "Intracerebral hemorrhage (ICH) is when blood suddenly bursts into brain tissue, causing damage to your brain. Symptoms usually appear suddenly during ICH. They include headache, weakness, confusion, and paralysis, particularly on one side of your body.",
    "Jaundice"                : "Yellow staining of the skin and sclerae (the whites of the eyes) by abnormally high blood levels of the bile pigment bilirubin. The yellowing extends to other tissues and body fluids. Jaundice was once called the morbus regius (the regal disease) in the belief that only the touch of a king could cure it",
    "Malaria"                 : "An infectious disease caused by protozoan parasites from the Plasmodium family that can be transmitted by the bite of the Anopheles mosquito or by a contaminated needle or transfusion. Falciparum malaria is the most deadly type.",
    "Chicken pox"             : "Chickenpox is a highly contagious disease caused by the varicella-zoster virus (VZV). It can cause an itchy, blister-like rash. The rash first appears on the chest, back, and face, and then spreads over the entire body, causing between 250 and 500 itchy blisters.",
    "Dengue"                  : "an acute infectious disease caused by a flavivirus (species Dengue virus of the genus Flavivirus), transmitted by aedes mosquitoes, and characterized by headache, severe joint pain, and a rash. â€” called also breakbone fever, dengue fever.",
    "Typhoid"                 : "An acute illness characterized by fever caused by infection with the bacterium Salmonella typhi. Typhoid fever has an insidious onset, with fever, headache, constipation, malaise, chills, and muscle pain. Diarrhea is uncommon, and vomiting is not usually severe.",
    "hepatitis A"             : "Hepatitis A is a highly contagious liver infection caused by the hepatitis A virus. The virus is one of several types of hepatitis viruses that cause inflammation and affect your liver's ability to function.",
    "Hepatitis B"             : "Hepatitis B is an infection of your liver. It can cause scarring of the organ, liver failure, and cancer. It can be fatal if it isn't treated. It's spread when people come in contact with the blood, open sores, or body fluids of someone who has the hepatitis B virus.",
    "Hepatitis C"             : "Inflammation of the liver due to the hepatitis C virus (HCV), which is usually spread via blood transfusion (rare), hemodialysis, and needle sticks. The damage hepatitis C does to the liver can lead to cirrhosis and its complications as well as cancer.",
    "Hepatitis D"             : "Hepatitis D, also known as the hepatitis delta virus, is an infection that causes the liver to become inflamed. This swelling can impair liver function and cause long-term liver problems, including liver scarring and cancer. The condition is caused by the hepatitis D virus (HDV).",
    "Hepatitis E"             : "A rare form of liver inflammation caused by infection with the hepatitis E virus (HEV). It is transmitted via food or drink handled by an infected person or through infected water supplies in areas where fecal matter may get into the water. Hepatitis E does not cause chronic liver disease.",
    "Alcoholic hepatitis"     : "Alcoholic hepatitis is a diseased, inflammatory condition of the liver caused by heavy alcohol consumption over an extended period of time. It's also aggravated by binge drinking and ongoing alcohol use. If you develop this condition, you must stop drinking alcohol",
    "Tuberculosis"            : "Tuberculosis (TB) is an infectious disease usually caused by Mycobacterium tuberculosis (MTB) bacteria. Tuberculosis generally affects the lungs, but can also affect other parts of the body. Most infections show no symptoms, in which case it is known as latent tuberculosis.",
    "Common Cold"             : "The common cold is a viral infection of your nose and throat (upper respiratory tract). It's usually harmless, although it might not feel that way. Many types of viruses can cause a common cold.",
    "Pneumonia"               : "Pneumonia is an infection in one or both lungs. Bacteria, viruses, and fungi cause it. The infection causes inflammation in the air sacs in your lungs, which are called alveoli. The alveoli fill with fluid or pus, making it difficult to breathe.",
    "Dimorphic hemmorhoids(piles)": "Hemorrhoids, also spelled haemorrhoids, are vascular structures in the anal canal. In their ... Other names, Haemorrhoids, piles, hemorrhoidal disease .",
    "Heart attack"            : "The death of heart muscle due to the loss of blood supply. The loss of blood supply is usually caused by a complete blockage of a coronary artery, one of the arteries that supplies blood to the heart muscle.",
    "Varicose veins"          : "A vein that has enlarged and twisted, often appearing as a bulging, blue blood vessel that is clearly visible through the skin. Varicose veins are most common in older adults, particularly women, and occur especially on the legs.",
    "Hypothyroidism"          : "Hypothyroidism, also called underactive thyroid or low thyroid, is a disorder of the endocrine system in which the thyroid gland does not produce enough thyroid hormone.",
    "Hyperthyroidism"         : "Hyperthyroidism (overactive thyroid) occurs when your thyroid gland produces too much of the hormone thyroxine. Hyperthyroidism can accelerate your body's metabolism, causing unintentional weight loss and a rapid or irregular heartbeat.",
    "Hypoglycemia"            : " Hypoglycemia is a condition in which your blood sugar (glucose) level is lower than normal. Glucose is your body's main energy source. Hypoglycemia is often related to diabetes treatment. But other drugs and a variety of conditions â€” many rare â€” can cause low blood sugar in people who don't have diabetes.",
    "Osteoarthritis"          : "Osteoarthritis is the most common form of arthritis, affecting millions of people worldwide. It occurs when the protective cartilage that cushions the ends of your bones wears down over time.",
    "Arthritis"               : "Arthritis is the swelling and tenderness of one or more of your joints. The main symptoms of arthritis are joint pain and stiffness, which typically worsen with age. The most common types of arthritis are osteoarthritis and rheumatoid arthritis.",
    "(vertigo) Paroxysmal Positional Vertigo": "Benign paroxysmal positional vertigo (BPPV) is one of the most common causes of vertigo â€” the sudden sensation that you're spinning or that the inside of your head is spinning. Benign paroxysmal positional vertigo causes brief episodes of mild to intense dizziness.",
    "Acne"                    : "Acne vulgaris is the formation of comedones, papules, pustules, nodules, and/or cysts as a result of obstruction and inflammation of pilosebaceous units (hair follicles and their accompanying sebaceous gland). Acne develops on the face and upper trunk. It most often affects adolescents.",
    "Urinary tract infection" : "Urinary tract infection: An infection of the kidney, ureter, bladder, or urethra. Abbreviated UTI. Not everyone with a UTI has symptoms, but common symptoms include a frequent urge to urinate and pain or burning when urinating.",
    "Psoriasis"               : "Psoriasis is a common skin disorder that forms thick, red, bumpy patches covered with silvery scales. They can pop up anywhere, but most appear on the scalp, elbows, knees, and lower back. Psoriasis can't be passed from person to person. It does sometimes happen in members of the same family.",
    "Impetigo"                : "Impetigo (im-puh-TIE-go) is a common and highly contagious skin infection that mainly affects infants and children. Impetigo usually appears as red sores on the face, especially around a child's nose and mouth, and on hands and feet. The sores burst and develop honey-colored crusts.",
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

    valid_symptoms = [s for s in symptoms if s in symptom_cols]
    if not valid_symptoms:
        return jsonify({'error': 'No valid symptoms recognized. Please select from the suggestions.'}), 400
    if len(valid_symptoms) < 2:
        return jsonify({'error': 'Please enter at least 2 valid symptoms for accurate prediction.'}), 400
    symptoms = valid_symptoms

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
            time.sleep(0.02)   # adjust speed here (seconds per word)
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

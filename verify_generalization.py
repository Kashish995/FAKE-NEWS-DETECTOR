import os
import sys
import pickle
import numpy as np
import re

PROJECT_DIR = r"c:\Users\swain\OneDrive\Desktop\FAKE-NEWS-DETECTION"
sys.path.append(PROJECT_DIR)

MODEL_PATH = os.path.join(PROJECT_DIR, "model", "model.pkl")
VECTORIZER_PATH = os.path.join(PROJECT_DIR, "model", "vectorizer.pkl")

# Source cleaning pattern
dateline_pattern = re.compile(r'^[A-Z][A-Za-z\.\s/]{2,40}\(Reuters\)\s*-\s*')
reuters_word_pattern = re.compile(r'\bReuters\b', re.IGNORECASE)

def clean_text(text):
    text = str(text)
    text = dateline_pattern.sub('', text, count=1)
    text = reuters_word_pattern.sub('', text)
    return text

test_cases = [
    {
        "topic": "India News",
        "expected": "Real",
        "text": "India's space agency ISRO successfully launched its latest Earth Observation Satellite EOS-08 from Sriharikota, achieving a precise orbit injection and completing another milestone in national space missions."
    },
    {
        "topic": "Sports News",
        "expected": "Real",
        "text": "Real Madrid secured their 15th UEFA Champions League title after defeating Borussia Dortmund 2-0 in the final at Wembley Stadium, with late goals from Dani Carvajal and Vinicius Junior sealing the victory."
    },
    {
        "topic": "Politics News",
        "expected": "Real",
        "text": "The United States Senate has passed a major bipartisan infrastructure bill to invest billions in highway repairs, public transport expansions, and clean water access across the country."
    },
    {
        "topic": "Tech News",
        "expected": "Real",
        "text": "Apple unveiled its new line of iPhones featuring a powerful A18 processor, enhanced camera sensors, and direct integration with artificial intelligence workflows during its annual launch event in Cupertino."
    },
    {
        "topic": "Fake News 1 (Conspiracy)",
        "expected": "Fake",
        "text": "Secret documents leaked from government archives prove that scientists have developed a cure for aging, but major pharmaceutical corporations are colluding with politicians to hide it from the public to protect their trillion-dollar healthcare profits."
    },
    {
        "topic": "Fake News 2 (Fabrication)",
        "expected": "Fake",
        "text": "Breaking news: A massive alien spaceship was spotted hovering over the Pacific Ocean yesterday, and government agencies have issued a complete media blackout to prevent widespread public panic."
    }
]

print("--- GENERALIZATION VERIFICATION ENGINE ---")
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    print("Error: Models do not exist. Please train the model first.")
    sys.exit(1)

clf = pickle.load(open(MODEL_PATH, "rb"))
vec = pickle.load(open(VECTORIZER_PATH, "rb"))

passed = 0
for tc in test_cases:
    cleaned = clean_text(tc["text"])
    X_vec = vec.transform([cleaned])
    pred = clf.predict(X_vec)[0]
    probs = clf.predict_proba(X_vec)[0]
    
    label = "Real" if pred == 1 else "Fake"
    conf = probs[1] if pred == 1 else probs[0]
    
    status = "PASS" if label == tc["expected"] else "FAIL"
    if status == "PASS":
        passed += 1
        
    print(f"[{status}] Topic: {tc['topic']}")
    print(f"  Expected: {tc['expected']}, Predicted: {label} (Confidence: {conf*100:.2f}%)")
    print(f"  Probabilities: Fake={probs[0]*100:.2f}%, Real={probs[1]*100:.2f}%\n")

print("=" * 45)
print(f"VERIFICATION SCORE: {passed}/{len(test_cases)} cases passed.")
print("=" * 45)

if passed == len(test_cases):
    sys.exit(0)
else:
    sys.exit(1)

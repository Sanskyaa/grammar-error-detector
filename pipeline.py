
import joblib
import spacy
import difflib
from transformers import T5Tokenizer, T5ForConditionalGeneration

# ---- Load models (runs once when imported) ----
clf = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
nlp = spacy.load("en_core_web_sm")
tokenizer = T5Tokenizer.from_pretrained("vennify/t5-base-grammar-correction")
model = T5ForConditionalGeneration.from_pretrained("vennify/t5-base-grammar-correction")

# ---- Correction function ----
def correct_sentence(sentence):
    input_text = "grammar: " + sentence
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=64, truncation=True)
    outputs = model.generate(inputs, max_length=64, num_beams=5, early_stopping=True)
    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected

# ---- Word diff ----
def get_word_diff(original, corrected):
    orig_words = original.split()
    corr_words = corrected.split()
    matcher = difflib.SequenceMatcher(None, orig_words, corr_words)
    diffs = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            diffs.append({
                "type": tag,
                "original_words": orig_words[i1:i2],
                "corrected_words": corr_words[j1:j2]
            })
    return diffs

# ---- Error type classifier ----
def classify_error(original, corrected):
    diffs = get_word_diff(original, corrected)
    if not diffs:
        return ["No significant change detected"]

    orig_doc = nlp(original)
    corr_doc = nlp(corrected)
    error_types = []
    verb_tags = {"VB", "VBD", "VBZ", "VBG", "VBN", "VBP"}

    for d in diffs:
        orig_tags = [t.tag_ for t in orig_doc if t.text in d["original_words"]]
        corr_tags = [t.tag_ for t in corr_doc if t.text in d["corrected_words"]]

        if any(t in verb_tags for t in orig_tags) or any(t in verb_tags for t in corr_tags):
            error_types.append("Tense / Verb Form Error")
        elif any(t == "DT" for t in orig_tags) or any(t == "DT" for t in corr_tags):
            error_types.append("Article Error")
        elif any(t == "IN" for t in orig_tags) or any(t == "IN" for t in corr_tags):
            error_types.append("Preposition Error")
        elif any(t in ("NN", "NNS", "PRP") for t in orig_tags) and any(t in verb_tags for t in corr_tags):
            error_types.append("Subject-Verb Agreement Error")
        else:
            error_types.append("Syntax / Word Choice Error")

    return list(set(error_types))

# ---- Full pipeline ----
def grammar_error_detector(sentence):
    vec = vectorizer.transform([sentence])
    pred = clf.predict(vec)[0]

    result = {
        "input_sentence": sentence,
        "has_error": bool(pred == 0),
        "error_types": [],
        "corrected_sentence": sentence
    }

    if pred == 0:
        corrected = correct_sentence(sentence)
        result["corrected_sentence"] = corrected
        if corrected.strip().lower() != sentence.strip().lower():
            result["error_types"] = classify_error(sentence, corrected)
        else:
            result["error_types"] = ["Detected as ungrammatical, but no correction change found"]

    return result

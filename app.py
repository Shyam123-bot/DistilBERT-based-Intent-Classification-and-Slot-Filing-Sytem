from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from transformers import DistilBertTokenizerFast
import numpy as np
import json
import os
from src.model import JointIntentAndSlotFillingModel

# ---- Load Tokenizer ----
tokenizer = DistilBertTokenizerFast.from_pretrained("saved_model/")

# ---- Load Label Maps ----
with open("saved_model/intent_labels.json") as f:
    intent_map = json.load(f)
intent_names = [intent_map[str(i)] for i in range(len(intent_map))]

with open("saved_model/slot_labels.json") as f:
    slot_map = json.load(f)
slot_names = [slot_map[str(i)] for i in range(len(slot_map))]

# ---- Load Model ----
model = JointIntentAndSlotFillingModel(
    intent_num_labels=len(intent_names),
    slot_num_labels=len(slot_names)
)

# ✅ Initialize model with dummy input to create variables
_ = model({
    "input_ids": tf.zeros((1, 32), dtype=tf.int32),
    "attention_mask": tf.ones((1, 32), dtype=tf.int32)
}, training=False)

# ✅ Load trained weights
model.load_weights("saved_model/joint_model_weights.h5")

# ---- FastAPI ----
app = FastAPI()

# ---- Request Schema ----
class NLURequest(BaseModel):
    text: str

# ---- Decode BIO Slots ----
def decode_predictions(text, slot_logits, intent_logits):
    tokens = tokenizer.tokenize(text)
    slot_ids = tf.argmax(slot_logits, axis=-1).numpy()[0][1:-1]  # skip CLS, SEP
    intent_id = tf.argmax(intent_logits, axis=-1).numpy()[0]
    confidence = tf.nn.softmax(intent_logits, axis=-1).numpy()[0][intent_id]

    if confidence < 0.5:
        return {
            "intent": "fallback",
            "slots": {},
            "message": "I'm not sure I understood that. Could you say it again?"
        }

    collected_slots = {}
    active_slot_name = None
    active_slot_words = []

    for word in text.split():
        token_count = len(tokenizer.tokenize(word))
        if len(slot_ids) == 0:
            break
        slot_index = slot_ids[0]
        slot_label = slot_names[slot_index] if slot_index < len(slot_names) else "O"
        slot_ids = slot_ids[token_count:]

        if slot_label == "O":
            if active_slot_name:
                collected_slots[active_slot_name] = " ".join(active_slot_words)
                active_slot_name = None
                active_slot_words = []
        else:
            label_type = slot_label[2:]  # Remove B-/I-
            if active_slot_name == label_type:
                active_slot_words.append(word)
            else:
                if active_slot_name:
                    collected_slots[active_slot_name] = " ".join(active_slot_words)
                active_slot_name = label_type
                active_slot_words = [word]

    if active_slot_name:
        collected_slots[active_slot_name] = " ".join(active_slot_words)

    return {
        "intent": intent_names[intent_id],
        "slots": collected_slots,
        "confidence": round(float(confidence), 4)
    }

# ---- Inference Endpoint ----
@app.post("/nlu")
def get_nlu(req: NLURequest):
    inputs = tokenizer(req.text, return_tensors="tf", padding=True, truncation=True, max_length=32)
    slot_logits, intent_logits = model(inputs, training=False)
    return decode_predictions(req.text, slot_logits, intent_logits)

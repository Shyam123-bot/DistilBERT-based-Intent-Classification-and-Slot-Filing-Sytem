import tensorflow as tf
from transformers import TFDistilBertModel
from tensorflow.keras.layers import Dropout, Dense
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.metrics import SparseCategoricalAccuracy

class JointIntentAndSlotFillingModel(tf.keras.Model):
    def __init__(self, intent_num_labels=None, slot_num_labels=None,
                 model_name="distilbert-base-uncased", dropout_prob=0.1):
        super().__init__(name="joint_intent_slot")
        self.bert = TFDistilBertModel.from_pretrained(model_name)
        self.dropout = Dropout(dropout_prob)
        self.intent_classifier = Dense(intent_num_labels, name="intent_classifier")
        self.slot_classifier = Dense(slot_num_labels, name="slot_classifier")

    def call(self, inputs, training=False, return_attentions=False):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            training=training,
            output_attentions=return_attentions  # 🔥 conditionally collect attention
        )

        sequence_output = outputs.last_hidden_state
        cls_output = self.dropout(sequence_output[:, 0, :], training=training)
        intent_logits = self.intent_classifier(cls_output)

        sequence_output = self.dropout(sequence_output, training=training)
        slot_logits = self.slot_classifier(sequence_output)

        if return_attentions:
            return slot_logits, intent_logits, outputs.attentions
        else:
            return slot_logits, intent_logits
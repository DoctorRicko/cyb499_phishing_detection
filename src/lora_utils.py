# src/lora_utils.py
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification

def prepare_lora_model(model_name, num_labels=2):
    # Base model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        torch_dtype="auto"
    )

    # LoRA configuration
    lora_config = LoraConfig(
        r=16,                  # Rank
        lora_alpha=32,         # Scaling factor
        target_modules=["q_proj", "v_proj"],  # Modules to adapt
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLASSIFICATION"
    )

    # Wrap model with LoRA
    return get_peft_model(model, lora_config)
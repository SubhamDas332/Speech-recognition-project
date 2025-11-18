import os
import json
import librosa
from datasets import load_from_disk
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Trainer,
    TrainingArguments
)

# -------------------------------
# Data collator for dynamic padding
# -------------------------------
class DataCollatorCTCWithPadding:
    def __init__(self, processor, padding=True):
        self.processor = processor
        self.padding = padding

    def __call__(self, features):
        model_input_name = self.processor.model_input_names[0]
        input_features = [{model_input_name: f[model_input_name]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

# -------------------------------
# Main training
# -------------------------------
def main():
    # --- Load dataset ---
    prepared_dataset = load_from_disk("prepared_dataset")

    # --- Create vocabulary ---
    all_text = " ".join([t.lower() for t in prepared_dataset["train"]["transcript"] if t])
    all_text += " " + " ".join([t.lower() for t in prepared_dataset["test"]["transcript"] if t])
    vocab_list = sorted(list(set(all_text)))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict["|"] = vocab_dict.pop(" ")
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open("vocab.json", "w") as f:
        json.dump(vocab_dict, f)

    # --- Processor and tokenizer ---
    tokenizer = Wav2Vec2CTCTokenizer(
        "vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "facebook/wav2vec2-base"
    )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    # --- Prepare dataset ---
    def prepare_dataset(batch):
        # Fix double geo path
        audio_path = batch["file"].replace("/geo/geo/", "/geo/")
        audio_array, sr = librosa.load(audio_path, sr=16_000)
        batch["input_values"] = processor(audio_array, sampling_rate=sr).input_values[0]
        batch["input_length"] = len(batch["input_values"])

        transcript = batch["transcript"] if batch["transcript"] else ""
        # v4 syntax: use as_target_processor
        with processor.as_target_processor():
            batch["labels"] = processor(transcript).input_ids
        return batch

    prepared_dataset = prepared_dataset.map(
        prepare_dataset,
        remove_columns=prepared_dataset.column_names["train"],
        num_proc=1  # reduce memory usage
    )

    # --- Model and data collator ---
    data_collator = DataCollatorCTCWithPadding(processor=processor)

    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base",
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )
    model.freeze_feature_extractor()  # v4 method
    model.gradient_checkpointing_enable()  # reduce memory usage

    # --- TrainingArguments ---
    training_args = TrainingArguments(
        output_dir="wav2vec2-esperanto",
        per_device_train_batch_size=2,       # small batch size
        gradient_accumulation_steps=8,       # accumulate to keep effective batch size ~16
        num_train_epochs=30,
        fp16=True,                           # mixed precision
        bf16=False,
        save_steps=400,
        logging_steps=400,
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=prepared_dataset["train"],
        eval_dataset=prepared_dataset["validation"],
        tokenizer=processor.feature_extractor,
        data_collator=data_collator
    )

    # --- Train and save ---
    trainer.train()
    trainer.save_model("wav2vec2-esperanto-trained")


if __name__ == "__main__":
    main()
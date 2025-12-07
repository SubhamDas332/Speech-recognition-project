Esperanto Gibberish ASR: Optimizing Low-Resource Speech Recognition

📌 Project Overview:

This project implements a robust Automatic Speech Recognition (ASR) system fine-tuned to transcribe "Esperanto Gibberish"—a dataset characterized by phonetically consistent but semantically meaningless speech.

The core challenge was training a high-performance model on limited hardware (single RTX 3080 10GB) while overcoming significant acoustic generalization hurdles. The final model achieves a Word Error Rate (WER) of <0.19, achieved through iterative architecture switching, aggressive data augmentation, and memory optimization techniques.

Key Technical Achievements

Resource-Constrained Training: Successfully fine-tuned the 1-Billion parameter wav2vec2-xls-r-1b model on a 10GB GPU using LoRA (Low-Rank Adaptation) and Gradient Checkpointing.

Plateau Breaking: Overcame a stubborn WER plateau at 0.30 by switching base models and implementing a custom audiomentations pipeline.

Custom Tokenizer: Built a dynamic character-level vocabulary extractor to handle the specific phonetic distribution of the dataset.

Architecture & Methodology:

1. Model Selection

We experimented with two primary architectures:

Baseline: facebook/wav2vec2-large-xlsr-53 (300M params).

Final Production: facebook/wav2vec2-xls-r-1b (1B params). The 1B model provided the necessary acoustic resolution to distinguish gibberish phonemes without reliance on a language model.

2. Memory Optimization (The 10GB Constraint)

To fit the 1B model into VRAM, the following optimizations were implemented:

LoRA (Low-Rank Adaptation): Trained only adapter layers (Rank 128, Alpha 256) instead of full fine-tuning.

Gradient Checkpointing: Traded compute for memory to reduce activation footprint.

Batch Size = 1 with Gradient Accumulation = 16: Simulated a batch size of 16 to stabilize CTC loss gradients without OOM errors.

3. Data Augmentation Pipeline

To generalize beyond the limited training samples, a custom pipeline was injected into the Dataset class:

Time Stretch: Randomly speeding up/slowing down speech (0.85x - 1.15x).

Pitch Shift: Simulating different vocal tract lengths (+/- 2 semitones).

Gaussian Noise: Improving robustness against silence/background artifacts.

SpecAugment: Aggressive time/feature masking during the forward pass.

📉 Performance Evolution

Experiment Phase

Model Architecture

Strategy

Best WER

Phase 1

XLS-R (300M)

Full Fine-Tuning

0.35

Phase 2

XLS-R (300M)

LoRA (Rank 64) + Linear Sched

0.29

Phase 3

XLS-R (1B)

LoRA (Rank 64) + Grad Accum

0.22

Phase 4

XLS-R (1B)

LoRA (Rank 128) + Augmentation

< 0.19

💻 Installation & Usage

Prerequisites

pip install torch torchaudio transformers peft pandas jiwer audiomentations


Inference

To transcribe new audio files using the saved LoRA adapters:

from inference import transcribe_file, load_model

# Load 1B model with trained adapters
model, processor, vocab = load_model(
    base_model="facebook/wav2vec2-xls-r-1b",
    checkpoint_path="xlsr_1b_gibberish_best",
    vocab_path="vocab.json"
)

transcription = transcribe_file("path/to/audio.wav", model, processor, vocab)
print(f"Transcription: {transcription}")


Training

To reproduce the training results:

python train_1b_optimized.py


📂 Repository Structure

.
├── train_1b_optimized.py    # Main training loop with Augmentation & LoRA
├── inference.py             # Inference script for single files or CSVs
├── vocab.json               # Generated vocabulary mapping
├── geo/                     # Dataset folder
│   ├── train.csv
│   └── dev.csv
└── xlsr_1b_gibberish_best/  # Saved LoRA Adapters


Challenges:

Language Models & Gibberish: Standard N-gram Language Models harmed performance because they tried to "correct" valid gibberish into real words. Pure acoustic decoding (Greedy) proved superior.

Scheduler Impact: Switching from Linear Decay to a Cosine Scheduler was crucial for squeezing out the final 2% accuracy improvement during the "polishing" phase.

The 3080 Limit: Training 1B models on consumer hardware is viable but requires strict adherence to batch_size=1 and aggressive gradient accumulation.

📜 License

This project is licensed under the MIT License.

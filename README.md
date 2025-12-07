# **🎙️ Esperanto Gibberish ASR: Low-Resource Optimization**

 Achieved good results (WER \< 0.19) on a phonetically consistent but semantically meaningless dataset by fine-tuning a **1-Billion parameter model** on a single consumer **RTX 3080 (10GB)**.

## **📌 Project Overview**

This project implements a robust Automatic Speech Recognition (ASR) system tailored for "Esperanto Gibberish"—a dataset characterized by strict phonetic consistency without semantic meaning.

The core engineering challenge was training a massive model (Wav2Vec2-XLS-R-1B) on highly constrained hardware. Through iterative architecture switching, Low-Rank Adaptation (LoRA), and a custom augmentation pipeline, the system overcame acoustic generalization hurdles where standard models failed.

## **📉 Performance Evolution**

We adopted an iterative engineering approach to break through performance plateaus.

| Phase | Model Architecture | Strategy | WER (Word Error Rate) |
| :---- | :---- | :---- | :---- |
| **Phase 1** | XLS-R (300M) | Full Fine-Tuning | 0.35 |
| **Phase 2** | XLS-R (300M) | LoRA (Rank 64\) \+ Linear Sched | 0.29 |
| **Phase 3** | XLS-R (1B) | LoRA (Rank 64\) \+ Grad Accum \+ Greedy decoder| 0.22 |
| **Phase 4** | **XLS-R (1B)** | **LoRA (Rank 128\) \+ Augmentation** \+ 4 gram decoder| **\< 0.19 🏆** |

## **🛠️ Methodology & Architecture**

### **1\. Model Selection Strategy**

We experimented with two primary architectures:

* **Baseline:** facebook/wav2vec2-large-xlsr-53 (300M params).  
* **Final Production:** facebook/wav2vec2-xls-r-1b (1B params).  
  * *Insight:* The 1B model provided the necessary acoustic resolution to distinguish gibberish phonemes purely on sound, reducing reliance on linguistic probability.

### **2\. Memory Optimization (The 10GB Constraint)**

Fitting a 1B parameter model into a 10GB RTX 3080 required strict optimization:

* **LoRA (Low-Rank Adaptation):** Trained only adapter layers (Rank 128, Alpha 256\) rather than updating all model weights.  
* **Gradient Checkpointing:** Traded compute speed for memory by not caching intermediate activations.  
* **Gradient Accumulation:** Used a physical batch size of 1 with accumulation steps of 16 to simulate a batch size of 16, stabilizing CTC loss gradients without OOM errors.

### **3\. Data Augmentation Pipeline**

To generalize beyond limited training samples, we injected a custom audiomentations pipeline:

* **Time Stretch:** Randomly speeding up/slowing down speech (0.85x \- 1.15x).  
* **Pitch Shift:** Simulating different vocal tract lengths (+/- 2 semitones).  
* **SpecAugment:** Aggressive time/feature masking during the forward pass.  
* **Gaussian Noise:** Improved robustness against silence artifacts.

## **Insights & Challenges**

### ** Approaches so far**
* We initially relied on pure acoustic decoding (Greedy Search) supported by the high-capacity 1B model
* Realized that this harms performance as greedy decoder cannot distinguish between words like "cat" and "kat" where one is a valid word.
* Thus we implemented N-gram Language Models (KenLM) with BeamSearch. WER performance dropper from 0.25 to 0.19

### **📉 Scheduler Impact**

Switching from a standard Linear Decay to a **Cosine Scheduler** was crucial for the final "polishing" phase, squeezing out the final 2% accuracy improvement.

### **🧱 Custom Tokenization**

Built a dynamic character-level vocabulary extractor to handle the specific phonetic distribution of the dataset, rather than using a pre-trained tokenizer.

## **💻 Installation & Usage**

### **Prerequisites**
```bash
pip install torch torchaudio transformers peft pandas jiwer audiomentations
```
### **Inference**

To transcribe new audio files using the saved LoRA adapters:
```bash
from inference import transcribe\_file, load\_model

\# Load 1B model with trained adapters  
model, processor, vocab \= load\_model(  
    base\_model="facebook/wav2vec2-xls-r-1b",  
    checkpoint\_path="xlsr\_1b\_gibberish\_best",  
    vocab\_path="vocab.json"  
)

transcription \= transcribe\_file("path/to/audio.wav", model, processor, vocab)  
print(f"Transcription: {transcription}")
```
### **Training Reproduction**

To reproduce the training results using the optimized pipeline:
```bash
python train\_1b\_optimized.py
```
## **📂 Repository Structure**

.  
├── train\_1b\_optimized.py    \# Main training loop with Augmentation & LoRA  
├── inference.py             \# Inference script for single files or CSVs  
├── vocab.json               \# Generated vocabulary mapping  
├── geo/                     \# Dataset folder  
│   ├── train.csv  
│   └── dev.csv  
└── xlsr\_1b\_gibberish\_best/  \# Saved LoRA Adapters

## **📜 License**

This project is licensed under the MIT License.

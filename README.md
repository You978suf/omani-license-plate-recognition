# Omani License Plate Recognition (ALPR)

A deep-learning system that detects and reads **Omani license plates** from images, including **Arabic-script** plates.

## 🎯 Overview

This project improves a **CRNN (Convolutional Recurrent Neural Network)** recognition pipeline by replacing the **ResNet18** backbone with a **Bidirectional LSTM + CTC Loss** architecture — boosting character-recognition accuracy on Arabic-script Omani plates. CTC loss handles variable-length plate sequences without needing character-level alignment.

## 🧠 Approach

```
Plate image ──► CNN feature extractor ──► BiLSTM (sequence modeling) ──► CTC decoder ──► plate text
```

- **Backbone:** CNN feature extractor (baseline used ResNet18; improved with a custom CNN + BiLSTM head)
- **Sequence head:** Bidirectional LSTM
- **Loss:** Connectionist Temporal Classification (CTC) — alignment-free, variable-length output
- Compared multiple strategies (see the notebooks) and tracked results in `strategies.json`.

## 📂 Repository

| File | Purpose |
|------|---------|
| `model.py` | Model architecture (CNN + BiLSTM + CTC) |
| `predict.py` | Run inference on plate images |
| `license_plate_models.ipynb` | Main modeling notebook |
| `cnn_transformer_ctc_clean.ipynb` | CNN + CTC experiment |
| `crnn_ctc_strategy3_draft1.ipynb` | CRNN + CTC strategy |
| `baseline_resnet_mlp2.ipynb` | ResNet18 baseline for comparison |
| `strategies.json` | Tracked experiment strategies |
| `requirements.txt` | Python dependencies |

> Datasets, archives, and trained weights (`*.zip`, `*.pt`, `train/`, `val/`) are excluded via `.gitignore` to keep the repo lightweight.

## 🚀 Usage

```bash
pip install -r requirements.txt
python predict.py --image path/to/plate.jpg
```
(Open the notebooks to reproduce training and evaluation.)

## 🛠️ Tech

Python · PyTorch · CRNN · Bidirectional LSTM · CTC Loss · OpenCV

---

*Course project — Sultan Qaboos University.*

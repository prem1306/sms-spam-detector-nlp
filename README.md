# SMS Spam Detection — NLP Mini Project

Classify an English SMS as **spam** (unwanted promotional/fraudulent messages) or **ham** (ordinary messages). The enhanced default model combines word and character TF-IDF features with Complement Naive Bayes and is trained with an external SMS-spam augmentation set.

## Quick start

Use Python 3.11 or newer. Open a terminal in this folder:

```sh
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux instead: source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py                 # original baseline
python train_enhanced.py        # enhanced default model; see DATA_SOURCES.md
python app.py
```

The desktop demo uses tkinter, normally included with Windows Python. On Linux it may require your distribution's python3-tk package. A GUI-free demo is also available:

```sh
python predict.py "Congratulations! Claim your free prize now. Call today!"
python predict.py "Can we meet at the library after class?"
```

The UCI dataset is included. Training takes place locally and writes `model.joblib` / `enhanced_model.joblib` plus JSON metrics. The desktop app prefers the enhanced model when available. See `DATA_SOURCES.md` before reproducing the external augmentation run. The demo sends no messages to external services.

## Files

- `train.py`: baseline dataset cleaning, stratified split, training, majority baseline, evaluation and saved model.
- `train_enhanced.py`: enhanced model using word + character n-grams and externally sourced spam augmentation.
- `predict.py`: reusable prediction functions and command-line demo.
- `app.py`: desktop interface.
- `research_paper.md` / `research_paper.docx`: the submitted research paper.
- `DATA_SOURCES.md`: data provenance, redistribution decision and reproducibility instructions.
- `viva.md`: short presentation outline and oral-exam preparation.
- `data/SMSSpamCollection`: original labelled SMS data.
- `data/readme`: original dataset documentation.
- `results/metrics.json`: actual metrics from the included training run.

## Dataset and attribution

Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84

Source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

Dataset license: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/. The included source data is unmodified. Cleaning is performed in memory during training. To restore missing data, download the ZIP from the UCI source page and place `SMSSpamCollection` and `readme` in `data/`.

## Submission preparation

Add your name, roll number and course details to the report. Review the code and viva notes, run the project, and follow your course's rules on external assistance and attribution.

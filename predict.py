"""Command-line demo: python predict.py "Your message here"."""
import argparse
from pathlib import Path
import joblib


def load_model():
    root = Path(__file__).resolve().parent
    path = root / 'enhanced_model.joblib'
    if not path.exists():
        path = root / 'model.joblib'
    if not path.exists():
        raise SystemExit('Run python train_enhanced.py (or python train.py) first.')
    # Load only the locally trained model; joblib files can execute Python code.
    return joblib.load(path)


def classify(model, message):
    if not message.strip():
        raise ValueError('Enter a nonempty message.')
    probabilities = model.predict_proba([message])[0]
    label = model.predict([message])[0]
    return str(label), float(probabilities[list(model.classes_).index('spam')])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify an SMS as spam or ham.')
    parser.add_argument('message')
    args = parser.parse_args()
    try:
        label, score = classify(load_model(), args.message)
    except ValueError as error:
        parser.error(str(error))
    print(f'Prediction: {label} | Model spam probability: {score:.1%}')

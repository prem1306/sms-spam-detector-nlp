"""Reproducible enhanced model: UCI SMS training set + external spam augmentation.

The external CSV is intentionally not committed because its mirror has no stated
redistribution licence.  Obtain it from the source link in DATA_SOURCES.md.
"""
import csv
import json
from pathlib import Path
import joblib
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parent
DEFAULT_EXTERNAL = ROOT.parent.parent / 'work' / 'indian-sms-source' / 'indian_spam.csv'


def normalized(text):
    return ' '.join(text.lower().split())


def read_uci():
    records, conflicts = {}, set()
    for line in (ROOT / 'data' / 'SMSSpamCollection').read_text(encoding='utf-8').splitlines():
        label, message = line.split('\t', 1)
        key = normalized(message)
        if key in records and records[key][1] != label:
            conflicts.add(key)
        records[key] = (message, label)
    return [v for k, v in records.items() if k not in conflicts]


def read_external_spam(path):
    messages = []
    with Path(path).open(encoding='utf-8', newline='') as file:
        for row in csv.DictReader(file):
            if row.get('v1', '').strip().lower() == 'spam' and row.get('v2', '').strip():
                messages.append(row['v2'].strip())
    return messages


def train(external_path=DEFAULT_EXTERNAL):
    rows = read_uci()
    texts, labels = zip(*rows)
    # Test set remains entirely UCI, held out before external augmentation.
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=42, stratify=labels)
    external = read_external_spam(external_path)
    test_keys = {normalized(x) for x in x_test}
    train_keys = {normalized(x) for x in x_train}
    external = [x for x in external if normalized(x) not in test_keys | train_keys]
    augmented_x = list(x_train) + external
    augmented_y = list(y_train) + ['spam'] * len(external)
    model = Pipeline([
        ('features', FeatureUnion([
            ('word', TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True,
                                     min_df=1, max_df=0.995)),
            ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True,
                                     min_df=2, max_features=75000)),
        ])),
        ('classifier', ComplementNB(alpha=0.25)),
    ])
    model.fit(augmented_x, augmented_y)
    predicted = model.predict(x_test)
    metrics = {
        'uci_clean_rows': len(rows), 'uci_train_rows': len(x_train), 'uci_test_rows': len(x_test),
        'external_spam_rows_used': len(external), 'random_state': 42,
        'sklearn_version': sklearn.__version__, 'accuracy': accuracy_score(y_test, predicted),
        'classification_report': classification_report(y_test, predicted, output_dict=True, zero_division=0),
        'confusion_matrix_labels': ['ham', 'spam'],
        'confusion_matrix': confusion_matrix(y_test, predicted, labels=['ham', 'spam']).tolist(),
    }
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results' / 'enhanced_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    joblib.dump(model, ROOT / 'enhanced_model.joblib')
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == '__main__':
    train()

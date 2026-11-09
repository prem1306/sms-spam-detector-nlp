"""Train and evaluate an SMS classifier without test-data leakage."""
import json
from pathlib import Path
import joblib
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent


def train():
    source = ROOT / 'data' / 'SMSSpamCollection'
    if not source.exists():
        raise SystemExit('Dataset missing. See README.md for the UCI download instructions.')
    # Collapse case/whitespace-equivalent messages BEFORE splitting to avoid duplicates
    # of the same message appearing in both training and test sets.
    records = {}
    raw_count = 0
    conflicts = set()
    for line in source.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        label, message = line.split('\t', 1)
        if label not in {'ham', 'spam'}:
            raise ValueError(f'Unexpected label: {label}')
        raw_count += 1
        key = ' '.join(message.lower().split())
        if key in records and records[key][1] != label:
            conflicts.add(key)
        records[key] = (message, label)
    rows = [value for key, value in records.items() if key not in conflicts]
    texts, labels = zip(*rows)
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels)
    model = Pipeline([
        ('tfidf', TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)),
        ('classifier', MultinomialNB(alpha=0.5)),
    ])
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    baseline = DummyClassifier(strategy='most_frequent').fit(x_train, y_train)
    results = {
        'raw_rows': raw_count, 'clean_rows': len(rows),
        'conflicting_message_groups_removed': len(conflicts),
        'train_rows': len(x_train), 'test_rows': len(x_test),
        'random_state': 42, 'sklearn_version': sklearn.__version__,
        'accuracy': accuracy_score(y_test, predicted),
        'majority_baseline_accuracy': accuracy_score(y_test, baseline.predict(x_test)),
        'classification_report': classification_report(y_test, predicted, output_dict=True, zero_division=0),
        'confusion_matrix_labels': ['ham', 'spam'],
        'confusion_matrix': confusion_matrix(y_test, predicted, labels=['ham', 'spam']).tolist(),
    }
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results' / 'metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    joblib.dump(model, ROOT / 'model.joblib')
    print(classification_report(y_test, predicted, digits=4, zero_division=0))
    print('Confusion matrix (rows = actual, columns = predicted; ham, spam):')
    print(results['confusion_matrix'])
    print(f'Majority baseline accuracy: {results["majority_baseline_accuracy"]:.4f}')
    return results


if __name__ == '__main__':
    train()

# Presentation and viva notes

## Five-slide outline

1. **Problem:** identify unwanted SMS messages using binary text classification.
2. **Dataset:** UCI SMS Spam Collection; ham/spam labels; remove equivalent duplicates and conflicting labels before splitting.
3. **Method:** text → word unigrams/bigrams → TF-IDF vectors → Multinomial Naive Bayes → prediction.
4. **Evaluation:** use the measured results in report.md; explain spam precision, recall, F1 and the confusion matrix; compare with the majority baseline.
5. **Demo and limitations:** classify a normal message and a promotional message; discuss slang, new scams and multilingual messages.

## Questions and answers

**What is NLP?** Methods for enabling computers to process and analyse human language.

**What is TF-IDF?** Term frequency measures usage in a document; inverse document frequency downweights terms that occur across many documents. Here the weighted vectors are L2-normalized.

**Why Naive Bayes?** It is fast and works well with sparse text vectors. It assumes features are conditionally independent given the class. With TF-IDF its multinomial likelihood is a useful approximation.

**What does alpha do?** Additive smoothing avoids zero likelihoods for features not observed in a class. This project uses a fixed alpha of 0.5 and does not tune on the test set.

**Why split the data?** To estimate performance on messages unseen during training. The split is stratified to preserve approximately the same class proportions.

**Why fit TF-IDF only on training data?** Learning vocabulary or IDF from test messages leaks information about the test set into training.

**Why can accuracy be misleading?** Ham is much more common. A classifier that always predicts ham can have high accuracy while missing every spam message.

**Precision versus recall?** Spam precision is the fraction of predicted spam that really is spam. Spam recall is the fraction of actual spam detected. F1 is their harmonic mean.

**False positive versus false negative?** A false positive flags legitimate ham as spam; a false negative lets spam through.

**Why no stemming or stop-word removal?** They are optional. Short messages may contain informative common words; this simple baseline keeps them and uses lowercasing and default word tokenization.

**Is the score a guarantee?** No. Naive Bayes probabilities are not calibrated here. Unknown vocabulary, language shifts and changing spam tactics can cause mistakes.

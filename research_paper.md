# SMS Spam Detection with External Spam Augmentation and Hybrid TF-IDF Features

## A comparative study of baseline and enhanced Naive Bayes classifiers

**K Prem Sai**  
**Registration number: 11249A148**  
**NLP Mini Project**  
**Date: 10 November 2026**

## Abstract

Short-message-service (SMS) spam remains a useful natural-language-processing problem because messages are brief, informal, noisy, and often intentionally deceptive. This project develops and evaluates two lightweight supervised classifiers for binary SMS spam detection. The baseline uses word-level term frequency–inverse document frequency (TF-IDF) features and Multinomial Naive Bayes. The enhanced system combines word unigram/bigram TF-IDF features with character 3–5 gram TF-IDF features, uses Complement Naive Bayes, and augments the training partition with 1,000 externally sourced messages labelled as spam. Evaluation is performed on the same strictly held-out UCI SMS Spam Collection test partition for both systems; no external message is used in test evaluation. The baseline reached 96.71% accuracy and 84.68% spam F1, with 73.44% spam recall. The enhanced model reached 98.06% accuracy and 92.25% spam F1, with 92.97% spam recall. This is an increase of 19.53 percentage points in spam recall, although false positives increased from 0 to 11. The results show that external spam augmentation and character features can substantially improve the detection of spam patterns, while also illustrating why accuracy alone is insufficient for evaluating an imbalanced spam filter.

**Keywords:** natural language processing, SMS spam, TF-IDF, Complement Naive Bayes, data augmentation, text classification.

## 1. Introduction

SMS is still used for service alerts, personal communication, authentication, and marketing. Its reach and brevity also make it attractive to fraudsters. Spam messages may promote unwanted services, persuade recipients to call premium-rate numbers, impersonate delivery companies or banks, or direct recipients to phishing links. A useful filter must recognize suspicious lexical patterns while avoiding the equally costly outcome of classifying a legitimate personal message as spam.

This task is a binary text-classification problem: given a message, assign either *ham* (legitimate) or *spam*. It is well suited to traditional machine learning because the input is short text and labelled examples are available. Although transformer models are widely used for NLP, a compact TF-IDF/Naive-Bayes model is a good mini-project choice. It trains quickly on a standard computer, its components are easy to explain, and it gives a strong transparent baseline.

The main question addressed here is whether a small conventional classifier can be improved responsibly by adding externally found spam examples and a richer representation of message text. The work therefore compares two systems under one fixed evaluation protocol. The first is the original word-TF-IDF Multinomial Naive Bayes system. The second adds external spam data to training and uses both word and character n-grams with Complement Naive Bayes. The test set is never augmented, ensuring that the reported comparison reflects generalization to untouched UCI messages rather than memorization.

## 2. Related background

The UCI SMS Spam Collection is a classic corpus for spam-filtering research. It contains 5,574 English SMS messages labelled as ham or spam and was assembled from several public or research sources [1]. Its labels and relatively small size make it suitable for a reproducible classroom experiment. The corpus is imbalanced: ham messages are substantially more common than spam. Consequently, a trivial classifier that always predicts ham can appear accurate while failing to detect the class that matters most.

TF-IDF represents each document as a sparse vector. For a term *t* in message *d*, term frequency describes its occurrence in that message, while inverse document frequency decreases the importance of terms that occur in many documents. A common formulation is:

`tfidf(t,d) = tf(t,d) × log((1 + N) / (1 + df(t))) + 1`.

Here, *N* is the number of training documents and `df(t)` is the number containing the term. This project uses the implementation in scikit-learn, with L2 normalization and sublinear term frequency [2]. Word unigrams capture individual tokens such as “free”, whereas word bigrams capture short phrases such as “claim now”. Character n-grams add resilience to unusual spellings, embedded punctuation, URLs, phone-like sequences, and creative obfuscation often used in spam.

Naive Bayes estimates a class posterior using class priors and feature likelihoods. Its simplifying conditional-independence assumption is not literally true for language, but it works well in high-dimensional sparse text classification. Multinomial Naive Bayes is a conventional baseline. Complement Naive Bayes estimates feature statistics from the complement of each class and is designed to be more robust under class imbalance [3]. Additive smoothing prevents a previously unseen feature from forcing a zero likelihood.

## 3. Objectives

This project has four objectives:

1. Build a transparent end-to-end NLP pipeline for SMS spam classification.
2. Compare a baseline classifier with an enhanced classifier trained with additional online spam examples.
3. Use a leak-resistant evaluation design and report class-specific metrics.
4. Deliver a local desktop demonstration that predicts spam or ham for a user-entered message.

## 4. Data and ethical handling

### 4.1 Base corpus

The included UCI file was parsed as tab-separated label/message pairs. During preprocessing, messages differing only in case and whitespace were collapsed before the train/test split. This reduces direct duplicate leakage. Groups with contradictory labels would be removed; none were found in this run. The process left 5,159 unique normalized-message groups.

An 80/20 stratified split with `random_state=42` produced 4,127 UCI training messages and 1,032 UCI test messages. Stratification preserves the approximate spam prevalence in each portion. The held-out test partition contains 904 ham and 128 spam messages.

### 4.2 External spam augmentation

For the enhanced experiment, 1,000 nonempty messages labelled `spam` were read from the publicly accessible `indian_spam.csv` mirror [4]. Its repository describes a crowdsourced Hindi-and-English SMS corpus attributed to Yadav et al. and reports participation by 43 people. Only its spam rows were added to the training side. Exact normalized overlaps with the UCI training and UCI test messages were removed before fitting; all 1,000 external spam messages remained after this check.

The source mirror does not state a redistribution licence. To respect that ambiguity, its raw rows are **not** committed to this repository. The paper and `DATA_SOURCES.md` identify the source and the project contains a reproducible training script that accepts a locally obtained copy. The learned enhanced model is included for demonstration, but users should verify the source terms before reproducing or republishing the external-data experiment. No attempt was made to identify or contact individuals represented in either corpus.

This decision has an important methodological limitation: the external messages may differ from the UCI corpus by region, language mixture, era, or annotation process. Such diversity can improve robustness but can also introduce a domain shift. Results below are therefore evidence on this held-out UCI partition, not a guarantee of present-day deployment performance.

## 5. Methodology

### 5.1 Baseline system

The baseline follows this pipeline:

1. Lowercase messages and tokenize into word unigrams and bigrams.
2. Learn TF-IDF vocabulary and weights from UCI training messages only.
3. Train `MultinomialNB(alpha=0.5)`.
4. Predict the unseen UCI test set.

The majority baseline, which always predicts ham, is also measured. It establishes a minimum meaningful comparison under the imbalanced label distribution.

### 5.2 Enhanced system

The enhanced system keeps the identical UCI test set and training/test split, then adds the external spam messages only to the training input. A `FeatureUnion` concatenates two independently fitted vector spaces:

* word TF-IDF: lowercased unigrams and bigrams, sublinear TF, `max_df=0.995`;
* character-within-word-boundaries TF-IDF: 3–5 grams, sublinear TF, `min_df=2`, capped at 75,000 features.

The combined representation is fitted using 4,127 UCI training messages plus 1,000 external spam messages. `ComplementNB(alpha=0.25)` is then trained. These settings were selected as a compact, explainable enhancement rather than by repeatedly tuning against the test split. The source code fixes the split seed and records the library version and metrics in JSON.

### 5.3 Evaluation measures

For the spam class, precision is `TP / (TP + FP)`, recall is `TP / (TP + FN)`, and F1 is the harmonic mean of precision and recall. A false positive is a legitimate message flagged as spam; a false negative is a spam message allowed through. Accuracy is `(TP + TN) / all messages`, but it is not sufficient on its own because of class imbalance. The confusion matrix gives the underlying counts and makes the operational tradeoff visible.

## 6. Results

Table 1 compares both learned systems on the same 1,032-message UCI test partition.

| Measure | Majority baseline | Word TF-IDF + Multinomial NB | Enhanced hybrid model |
|---|---:|---:|---:|
| Accuracy | 87.60% | 96.71% | 98.06% |
| Spam precision | 0.00% | 100.00% | 91.54% |
| Spam recall | 0.00% | 73.44% | 92.97% |
| Spam F1 | 0.00% | 84.68% | 92.25% |
| Ham recall | 100.00% | 100.00% | 98.78% |

The baseline model confusion matrix is shown below, with rows as actual class and columns as predicted class.

| Baseline actual / predicted | Ham | Spam |
|---|---:|---:|
| Ham | 904 | 0 |
| Spam | 34 | 94 |

The enhanced model confusion matrix is:

| Enhanced actual / predicted | Ham | Spam |
|---|---:|---:|
| Ham | 893 | 11 |
| Spam | 9 | 119 |

Compared with the baseline, the enhanced system catches 25 additional spam messages (119 versus 94) and reduces false negatives from 34 to 9. It introduces 11 false positives, where the baseline introduced none. Accuracy rises by 1.36 percentage points and spam F1 rises by 7.57 points. The strongest improvement is recall: 92.97% of spam messages were identified rather than 73.44%.

The result is plausible: the extra spam corpus gives the classifier more examples of promotional, deceptive, and informal patterns, while character features can recognize pieces of tokens even when words differ slightly. However, the increment in false positives shows the expected cost of making the classifier more sensitive to spam. In a real product, an organization would choose a decision threshold based on the relative harm of missed spam and incorrectly filtered legitimate messages. This mini-project retains the default class decision rule for clarity and does not claim that the chosen operating point is optimal.

## 7. Discussion

The experiment demonstrates that better feature coverage and augmentation can improve a small text classifier substantially. The improvement is not merely accuracy: spam recall and F1 also improve. The majority baseline helps interpret this result. Although it obtains 87.60% accuracy, it has zero spam recall and zero spam F1; it is unusable as a spam detector. A report that presented accuracy alone could therefore create a misleading impression.

The baseline was unusually conservative on this particular test split: it made no false-positive errors but missed 34 spam messages. The enhanced classifier shifts the balance. It becomes more willing to label a message spam, catching most previously missed examples but flagging 11 legitimate messages. The appropriate balance depends on context. A personal inbox might quarantine suspected messages for review, making false positives less damaging. A transactional alert system might prioritize ham recall more strongly.

Character n-grams are particularly appropriate for SMS. Message text commonly contains repeated punctuation, compressed spellings, currency symbols, URLs, phone numbers, and obfuscated words. Pure word tokenization can treat a minor change as a completely unseen token, whereas character features retain partial overlap. This is not a replacement for semantic understanding; rather, it is a cost-effective way to capture surface-form similarity.

External augmentation introduces several cautions. The additional messages are not a random sample of the target users’ incoming SMS traffic. Their Hindi/English composition and collection setting differ from UCI. The test corpus is also old and cannot represent modern smishing techniques such as QR-code lures, branded sender IDs, or generative language. A single held-out split also has sampling variation. Cross-validation can support model-selection decisions on training data, but a final untouched evaluation set should still be reserved. These constraints mean the results should be treated as an academic comparison, not a deployment certificate.

## 8. Reproducibility and software artifact

The repository contains `train.py` for the baseline and `train_enhanced.py` for the enhanced experiment. Both write their exact evaluation output to `results/` and use a fixed seed. `predict.py` loads the enhanced model by default, falling back to the baseline if necessary. `app.py` provides a tkinter desktop interface with no external network calls.

To reproduce the baseline, install the listed dependencies and run `python train.py`. To reproduce the enhanced system, obtain the external source under the applicable terms, place the CSV as described in `DATA_SOURCES.md`, and run `python train_enhanced.py`. The raw external data is intentionally absent from the public repository. This separation makes the provenance and redistribution boundary explicit rather than concealing it.

## 9. Conclusion

This NLP mini-project built a transparent SMS spam detector and compared a standard baseline with an enhanced version trained using external spam examples. On the same held-out UCI partition, the enhanced hybrid TF-IDF/Complement-Naive-Bayes system achieved 98.06% accuracy, 92.97% spam recall, and 92.25% spam F1. It found 25 more spam messages than the baseline but created 11 false positives. The result supports the use of data augmentation and character features for short noisy messages, while emphasizing a core lesson of text-classification evaluation: model quality must be assessed with precision, recall, F1, confusion matrices, source provenance, and test-set isolation—not accuracy alone.

## References

[1] T. Almeida and J. Hidalgo, “SMS Spam Collection,” UCI Machine Learning Repository, 2011. DOI: https://doi.org/10.24432/C5CC84.

[2] Scikit-learn developers, “Text feature extraction,” scikit-learn documentation. https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction (accessed 10 November 2026).

[3] Scikit-learn developers, “Naive Bayes,” scikit-learn documentation. https://scikit-learn.org/stable/modules/naive_bayes.html (accessed 10 November 2026).

[4] Prince Bari, “SMS Spam Classification on Indian Dataset: A Crowdsourced Collection of Hindi and English Messages,” GitHub repository. https://github.com/princebari/-SMS-Spam-Classification-on-Indian-Dataset-A-Crowdsourced-Collection-of-Hindi-and-English-Messages (accessed 10 November 2026).

[5] K. Yadav, P. Kumaraguru, A. Goyal, A. Gupta, and A. Naik, “SMSAssassin: Crowdsourcing driven mobile-based system for SMS spam filtering,” 2011. Source attribution supplied by [4].

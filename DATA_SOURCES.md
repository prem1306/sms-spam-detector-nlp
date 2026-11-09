# Data sources and reproducibility

## Included UCI corpus

The repository includes the original `SMSSpamCollection` file from Almeida and Hidalgo's UCI SMS Spam Collection. Cite: Almeida, T. & Hidalgo, J. (2011), DOI: https://doi.org/10.24432/C5CC84. UCI lists it under CC BY 4.0.

## External training augmentation (not redistributed)

`train_enhanced.py` uses 1,000 messages labelled `spam` from the `indian_spam.csv` mirror at https://github.com/princebari/-SMS-Spam-Classification-on-Indian-Dataset-A-Crowdsourced-Collection-of-Hindi-and-English-Messages. Its README attributes the collection to Yadav, Kumaraguru, Goyal, Gupta and Naik (2011), a crowdsourced Hindi/English SMS corpus from 43 participants.

The mirror does not state a redistribution licence. Therefore its raw messages are deliberately excluded from this public repository. To reproduce the enhanced run, obtain the data from the linked source only if your intended use permits it, then place `indian_spam.csv` at `../../work/indian-sms-source/indian_spam.csv` relative to this project, or edit `DEFAULT_EXTERNAL` in `train_enhanced.py`.

Only rows labelled `spam` are used. Exact case-and-whitespace normalized overlap with UCI training and test text is removed. The UCI test partition is created before augmentation and is never used to fit vectorizers, select parameters, or train the classifier.

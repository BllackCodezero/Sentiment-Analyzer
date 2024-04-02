# Sentiment Analysis Model for Twitter Data

A sentiment analysis model trained on a Twitter dataset to classify tweets as positive or negative.

## Dataset Information

- **Source**: Kaggle Twitter dataset
- **Size**: 10,000 tweets
- **Features**: Text, Sentiment Label (Positive, Negative)

## Features

- TF-IDF Vectorization

## Requirements

- Python 3.x
- numpy
- pandas
- re
- nltk
- scikit-learn

## Dependencies

```python
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


Usage
To load the trained sentiment analysis model, follow these steps:
from sklearn.externals import joblib

# Load the trained model
model = joblib.load('model.sav')


Making Predictions
After loading the model, you can make predictions on new tweets as follows:
# Example tweet
tweet = "I love this product!"

# Predict sentiment
prediction = model.predict([tweet])

if prediction[0] == 0:
    print('Negative Tweet')
else:
    print('Positive Tweet')


Example Usage in a Python Script
Here's an example Python script demonstrating how to load the model and make predictions:
from sklearn.externals import joblib

# Load the trained model
model = joblib.load('model.sav')

def predict_sentiment(tweet):
    prediction = model.predict([tweet])
    if prediction[0] == 0:
        return 'Negative Tweet'
    else:
        return 'Positive Tweet'

# Example usage
tweet1 = "I love this product!"
tweet2 = "This product is terrible."

print(predict_sentiment(tweet1))  # Output: Positive Tweet
print(predict_sentiment(tweet2))  # Output: Negative Tweet


License
This project is licensed under the MIT License.

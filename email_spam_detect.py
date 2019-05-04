import pandas as pd
import numpy as np
import os
import email
import email.policy
from bs4 import BeautifulSoup
from nltk import stem
from nltk.corpus import stopwords

stemmer = stem.SnowballStemmer('english')
stopwords = set(stopwords.words('english'))

os.listdir('input/')


def load_email(directory, filename):
    with open(os.path.join(directory, filename), "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)


def html_to_plain(email):
    try:
        soup = BeautifulSoup(email.get_content(), 'html.parser')
        return soup.text.replace('\n\n','')
    except:
        return "empty"


# Removes stopwords, and applies stemmer to words
def sanitize_email(email_text):
    email_text = html_to_plain(email_text).lower()

    email_text = [word for word in email_text.split() if word not in stopwords]
    email_text = " ".join([stemmer.stem(word) for word in email_text])
    return email_text


ham_test_filenames = [name for name in sorted(os.listdir('input/ham_test')) if len(name) > 20]
ham_train_filenames = [name for name in sorted(os.listdir('input/ham_training')) if len(name) > 20]
spam_test_filenames = [name for name in sorted(os.listdir('input/spam_test')) if len(name) > 20]
spam_train_filenames = [name for name in sorted(os.listdir('input/spam_training')) if len(name) > 20]


ham_test_emails = [load_email('input/ham_test', filename=name) for name in ham_test_filenames]
ham_train_emails = [load_email('input/ham_training', filename=name) for name in ham_train_filenames]
spam_test_emails = [load_email('input/spam_test', filename=name) for name in spam_test_filenames]
spam_train_emails = [load_email('input/spam_training', filename=name) for name in spam_train_filenames]

ham_test_emails = [sanitize_email(email) for email in ham_test_emails]

print(ham_train_emails[3].get_content())

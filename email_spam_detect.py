import os
import email
import email.policy
from bs4 import BeautifulSoup
from collections import Counter
from nltk import stem
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import svm

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
    if email_text is None:
        return ''
    email_text = email_text.lower()

    email_text = [word for word in email_text.split() if word not in stopwords]
    email_text = " ".join([stemmer.stem(word) for word in email_text])
    return email_text


def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        return "multipart({})".format(", ".join([
            get_email_structure(sub_email)
            for sub_email in payload
        ]))
    else:
        return email.get_content_type()


def structures_counter(emails):
    structures = Counter()
    for email in emails:
        structure = get_email_structure(email)
        structures[structure] += 1
    return structures


def email_to_plain(email):
    for part in email.walk():
        part_content_type = part.get_content_type()
        if part_content_type not in ['text/plain','text/html']:
            continue
        try:
            part_content = part.get_content()
        except: # in case of encoding issues
            part_content = str(part.get_payload())
        if part_content_type == 'text/plain':
            return part_content
        else:
            return html_to_plain(part)


ham_test_filenames = [name for name in sorted(os.listdir('input/ham_test')) if len(name) > 20]
ham_train_filenames = [name for name in sorted(os.listdir('input/ham_training')) if len(name) > 20]
spam_test_filenames = [name for name in sorted(os.listdir('input/spam_test')) if len(name) > 20]
spam_train_filenames = [name for name in sorted(os.listdir('input/spam_training')) if len(name) > 20]


ham_test_emails = [load_email('input/ham_test', filename=name) for name in ham_test_filenames]
ham_train_emails = [load_email('input/ham_training', filename=name) for name in ham_train_filenames]
spam_test_emails = [load_email('input/spam_test', filename=name) for name in spam_test_filenames]
spam_train_emails = [load_email('input/spam_training', filename=name) for name in spam_train_filenames]


ham_train_emails = [sanitize_email(email_to_plain(email)) for email in ham_train_emails]
ham_test_emails = [sanitize_email(email_to_plain(email)) for email in ham_test_emails]
spam_train_emails = [sanitize_email(email_to_plain(email)) for email in spam_train_emails]
spam_test_emails = [sanitize_email(email_to_plain(email)) for email in spam_test_emails]

vectorizer = TfidfVectorizer()

ham_train_emails = vectorizer.fit_transform(ham_train_emails)
ham_test_emails = vectorizer.fit_transform(ham_test_emails)
spam_train_emails = vectorizer.fit_transform(spam_train_emails)
spam_test_emails = vectorizer.fit_transform(spam_test_emails)

svm = svm.SVC(C=1000)
svm.fit(ham_train_emails, spam_train_emails)

X_test = vectorizer.transform(ham_test_emails)
y_pred = svm.predict(X_test)
print(confusion_matrix(spam_test_emails, y_pred))

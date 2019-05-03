import pandas as pd
import numpy as np
import os
import email
import email.policy
from bs4 import BeautifulSoup

os.listdir('input/')

ham_test_filenames = [name for name in sorted(os.listdir('input/ham_test')) if len(name) > 20]
ham_train_filenames = [name for name in sorted(os.listdir('input/ham_training')) if len(name) > 20]
spam_test_filenames = [name for name in sorted(os.listdir('input/spam_test')) if len(name) > 20]
spam_train_filenames = [name for name in sorted(os.listdir('input/spam_training')) if len(name) > 20]

print(len(ham_test_filenames))
print(len(ham_train_filenames))
print(len(spam_test_filenames))
print(len(spam_test_filenames))

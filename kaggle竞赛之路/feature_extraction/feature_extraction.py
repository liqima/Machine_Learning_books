from sklearn.datasets import fetch_20newsgroups

news = fetch_20newsgroups(subset = 'all')


# split
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)


# feature extraction count
from sklearn.feature_extraction.text import CountVectorizer
count_vec = CountVectorizer(analyzer = 'word', stop_words = 'english')
x_count_train = count_vec.fit_transform(x_train)
x_count_test = count_vec.transform(x_test)

# feature extraction tfidf
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer(analyzer = 'word', stop_words = 'english')
x_tfidf_train = tfidf_vec.fit_transform(x_train)
x_tfidf_test = tfidf_vec.transform(x_test)

# bayes model
from sklearn.naive_bayes import MultinomialNB
mnb_count = MultinomialNB()
mnb_count.fit(x_count_train, y_train)
y_count_pred = mnb_count.predict(x_count_test)

mnb_tfidf = MultinomialNB()
mnb_tfidf.fit (x_tfidf_train, y_train)
y_tfidf_pred = mnb_tfidf.predict(x_tfidf_test)

# report
from sklearn.metrics import classification_report
print('accuracy with count is ', mnb_count.score(x_count_test, y_test))
print(classification_report(y_test, y_count_pred, target_names=news.target_names))

print('accuracy with tfidf is ', mnb_tfidf.score(x_tfidf_test, y_test))
print(classification_report(y_test, y_tfidf_pred, target_names=news.target_names))
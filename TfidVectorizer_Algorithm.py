import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier,LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from matplotlib import pyplot
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
class traintfid:
    df = pd.read_csv('dataset.csv', header= 0, encoding= 'unicode_escape')
    df = df.set_index('Sr_Num')
    y = df.label
    df = df.drop('label', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.2, random_state=42)

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform((X_train).values.astype('U'))
    tfidf_test = tfidf_vectorizer.transform((X_test).values.astype('U'))

    tfidf_vectorizer.get_feature_names_out()[-10:]

    tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())

    models = []
    models.append(('PAC',PassiveAggressiveClassifier(max_iter=500)))
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=500)))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', MultinomialNB()))

    results = []
    names = []
    for name, model in models:
        model.fit(tfidf_train, y_train)
        pred = model.predict(tfidf_test)
        score = accuracy_score(y_test, pred)
        cm = confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
        print("{0} : Confusion Matrix {1}".format(name,cm))
        print('%s: %f' % (name, score))
        print("***************************")
        names.append(name)
        results.append(score)

    pyplot.ylim(.100, .999)
    pyplot.bar(names, results, color ='maroon', width = 0.6)

    pyplot.title('Algorithm Comparison')
    pyplot.show()





# Created by: PyQt5 UI code generator 5.15.6
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from PyQt5.QtWidgets import QMessageBox
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split



class Ui_training(object):
    def setupUi(self, training):
        training.setObjectName("training")
        training.resize(846, 579)
        self.label = QtWidgets.QLabel(training)
        self.label.setGeometry(QtCore.QRect(90, 30, 711, 91))
        font = QtGui.QFont()
        font.setFamily("Nirmala UI")
        font.setPointSize(26)
        font.setBold(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.inputtext = QtWidgets.QTextEdit(training)
        self.inputtext.setGeometry(QtCore.QRect(430, 210, 361, 141))
        self.inputtext.setObjectName("inputtext")
        self.result = QtWidgets.QLabel(training)
        self.result.setGeometry(QtCore.QRect(270, 420, 151, 61))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.result.setFont(font)
        self.result.setScaledContents(True)
        self.result.setAlignment(QtCore.Qt.AlignCenter)
        self.result.setObjectName("result")
        self.traintfid = QtWidgets.QPushButton(training)
        self.traintfid.setGeometry(QtCore.QRect(40, 190, 131, 61))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.traintfid.setFont(font)
        self.traintfid.setObjectName("traintf")
        self.pushButton_c = QtWidgets.QPushButton(training)
        self.pushButton_c.setGeometry(QtCore.QRect(140, 330, 131, 61))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton_c.setFont(font)
        self.pushButton_c.setObjectName("pushButton_c")
        self.inputtext_2 = QtWidgets.QTextEdit(training)
        self.inputtext_2.setGeometry(QtCore.QRect(430, 400, 361, 87))
        self.inputtext_2.setObjectName("inputtext_2")
        self.traincountvec = QtWidgets.QPushButton(training)
        self.traincountvec.setGeometry(QtCore.QRect(220, 190, 131, 61))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.traincountvec.setFont(font)
        self.traincountvec.setObjectName("traincv")
        self.label_2 = QtWidgets.QLabel(training)
        self.label_2.setGeometry(QtCore.QRect(580, 170, 121, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")

        self.retranslateUi(training)
        QtCore.QMetaObject.connectSlotsByName(training)

        self.traintfid.clicked.connect(self.traintf)
        self.traincountvec.clicked.connect(self.traincv)
        self.pushButton_c.clicked.connect(self.classifyFunction)

    def retranslateUi(self, training):
        _translate = QtCore.QCoreApplication.translate
        training.setWindowTitle(_translate("training", "Dialog"))
        self.label.setText(_translate("training", "Fake News Detection"))
        self.result.setText(_translate("training", "Result"))
        self.traintfid.setText(_translate("training", "Training-tfid"))
        self.pushButton_c.setText(_translate("training", "Classify"))
        self.traincountvec.setText(_translate("training", "Training-countVec"))
        self.label_2.setText(_translate("training", "Input Text"))

    def traintf(self):
        from TfidVectorizer_Algorithm import traintfid
        traintfid()
        print("training completed with tfid vectorizer")

    def traincv(self):
        from CountVectorizer import traincv
        traincv()
        print("training completed with count vectorizer")

    def classifyFunction(self, pred=None):
        mytext = self.inputtext.toPlainText()
        print(mytext)
        self.inputtext.setPlainText(mytext)
        df = pd.read_csv('dataset.csv', header= 0,encoding= 'unicode_escape')
        df = df.set_index('Sr_Num')
        y = df.label
        df = df.drop('label', axis=1)
        df = df['text']
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        tfidf_train = tfidf_vectorizer.fit_transform(X_train)
        tfidf_test = tfidf_vectorizer.transform(X_test)
        tfidf_vectorizer.get_feature_names_out()[-10:]
        tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())
        model = PassiveAggressiveClassifier(max_iter=500)
        model.fit(tfidf_train, y_train)
        pred = model.predict(tfidf_test)
        score = accuracy_score(y_test, pred)
        cm = confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
        print("Confusion Matrix {0}".format(cm))
        print('Accuracy Score: %f' % (score))
        print("***************************")
        tfidf_test = tfidf_vectorizer.transform([mytext])
        pred = model.predict(tfidf_test)
        pred
        print(pred)
        a = (str(pred))

        self.inputtext_2.setText(a)




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    training = QtWidgets.QDialog()
    ui = Ui_training()
    ui.setupUi(training)
    training.show()
    sys.exit(app.exec_())

#Multiple Linear regression

 from sklearn.linear_model import LinearRegression
 X = [[6, 2], [8, 1], [10, 0], [14, 2], [18, 0]]
 y = [[7], [9], [13], [17.5], [18]]
 model = LinearRegression()
 model.fit(X, y)
 X_test = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]]
 y_test = [[11], [8.5], [15], [18], [11]]
 predictions = model.predict(X_test)
 for i, prediction in enumerate(predictions):
 print 'Predicted: %s, Target: %s' % (prediction, y_test[i])
 print 'R-squared: %.2f' % model.score(X_test, y_test)
Predicted: [ 10.0625], Target: [11]
Predicted: [ 10.28125], Target: [8.5]
Predicted: [ 13.09375], Target: [15]
Predicted: [ 18.14583333], Target: [18]
Predicted: [ 13.3125], Target: [11]
R-squared: 0.77


#Polynomial Regression

 import numpy as np
 import matplotlib.pyplot as plt
 from sklearn.linear_model import LinearRegression
 from sklearn.preprocessing import PolynomialFeatures
 X_train = [[6], [8], [10], [14], [18]]
 y_train = [[7], [9], [13], [17.5], [18]]
 X_test = [[6], [8], [11], [16]]
 y_test = [[8], [12], [15], [18]]
 regressor = LinearRegression()
 regressor.fit(X_train, y_train)
 xx = np.linspace(0, 26, 100)
 yy = regressor.predict(xx.reshape(xx.shape[0], 1))
 plt.plot(xx, yy)
 quadratic_featurizer = PolynomialFeatures(degree=2)
 X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
 X_test_quadratic = quadratic_featurizer.transform(X_test)
 regressor_quadratic = LinearRegression()
 regressor_quadratic.fit(X_train_quadratic, y_train)
 xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.
shape[0], 1))
 plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c='r',
linestyle='--')
 plt.title('Pizza price regressed on diameter')
 plt.xlabel('Diameter in inches')
 plt.ylabel('Price in dollars')
 plt.axis([0, 25, 0, 25])
 plt.grid(True)
 plt.scatter(X_train, y_train)
 plt.show()
 print X_train
 print X_train_quadratic
 print X_test
 print X_test_quadratic
 print 'Simple linear regression r-squared', regressor.score(X_
test, y_test)
 print 'Quadratic regression r-squared', regressor_quadratic.
score(X_test_quadratic, y_test)


#Cross Validation
>>> import pandas as pd
>>> from sklearn. cross_validation import cross_val_score
>>> from sklearn.linear_model import LinearRegression
>>> df = pd.read_csv('data/winequality-red.csv', sep=';')
>>> X = df[list(df.columns)[:-1]]
>>> y = df['quality']
>>> regressor = LinearRegression()
>>> scores = cross_val_score(regressor, X, y, cv=5)
>>> print scores.mean(), scores
0.290041628842 [ 0.13200871 0.31858135 0.34955348 0.369145
0.2809196 ]

#Stochastic Gradient Descent
>>> import numpy as np
>>> from sklearn.datasets import load_boston
>>> from sklearn.linear_model import SGDRegressor
>>> from sklearn.cross_validation import cross_val_score
>>> from sklearn.preprocessing import StandardScaler
>>> from sklearn.cross_validation import train_test_split
>>> data = load_boston()
>>> X_train, X_test, y_train, y_test = train_test_split(data.data,
data.target)

>>> X_scaler = StandardScaler()
>>> y_scaler = StandardScaler()
>>> X_train = X_scaler.fit_transform(X_train)
>>> y_train = y_scaler.fit_transform(y_train)
>>> X_test = X_scaler.transform(X_test)
>>> y_test = y_scaler.transform(y_test)

>>> regressor = SGDRegressor(loss='squared_loss')
>>> scores = cross_val_score(regressor, X_train, y_train, cv=5)
>>> print 'Cross validation r-squared scores:', scores
>>> print 'Average cross validation r-squared score:', np.mean(scores)
>>> regressor.fit_transform(X_train, y_train)
>>> print 'Test set r-squared score', regressor.score(X_test, y_test)

Cross validation r-squared scores: [ 0.73428974 0.80517755
0.58608421 0.83274059 0.69279604]
Average cross validation r-squared score: 0.730217627242
Test set r-squared score 0.653188093125

#Feature Extraction and PreProcessing

#One Hot Encoding
>>> from sklearn.feature_extraction import DictVectorizer
>>> onehot_encoder = DictVectorizer()
>>> instances = [
>>> {'city': 'New York'},
>>> {'city': 'San Francisco'},
>>> {'city': 'Chapel Hill'}>>> ]
>>> print onehot_encoder.fit_transform(instances).toarray()
[[ 0. 1. 0.] [ 0. 0. 1.][ 1. 0. 0.]]

#Corpus | Feature Vector | Total words in corpus each given index if word found in Document assigned 1 or else 0 

{u'duke': 2, u'basketball': 1, u'lost': 5, u'played': 6, u'in': 4,
u'game': 3, u'sandwich': 7, u'unc': 9, u'ate': 0, u'the': 8}
Now, our feature vectors are as follows:
UNC played Duke in basketball = [[0 1 1 0 1 0 1 0 0 1]]
Duke lost the basketball game = [[0 1 1 1 0 1 0 0 1 0]]
I ate a sandwich = [[1 0 0 0 0 0 0 1 0 0]]

#Remove Stop Words
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> corpus = [
>>> 'UNC played Duke in basketball',
>>> 'Duke lost the basketball game',
>>> 'I ate a sandwich'
>>> ]
>>> vectorizer = CountVectorizer(stop_words='english')
>>> print vectorizer.fit_transform(corpus).todense()
>>> print vectorizer.vocabulary_
[[0 1 1 0 0 1 0 1]
[0 1 1 1 1 0 0 0]
[1 0 0 0 0 0 1 0]]
{u'duke': 2, u'basketball': 1, u'lost': 4, u'played': 5, u'game': 3,
u'sandwich': 6, u'unc': 7, u'ate': 0}



#lemmatization And Stemmer | Dimensionality Reduction
corpus = [
'I am gathering ingredients for the sandwich.',
'There were many wizards at the gathering.'
]
>>> import nltk
>>> nltk.download()

>>> from nltk.stem.wordnet import WordNetLemmatizer
>>> lemmatizer = WordNetLemmatizer()
>>> print lemmatizer.lemmatize('gathering', 'v')
>>> print lemmatizer.lemmatize('gathering', 'n')
gather
gathering

>>> from nltk.stem import PorterStemmer
>>> stemmer = PorterStemmer()
>>> print stemmer.stem('gathering')
gather

>>> from nltk import word_tokenize
>>> from nltk.stem import PorterStemmer
>>> from nltk.stem.wordnet import WordNetLemmatizer
>>> from nltk import pos_tag
>>> wordnet_tags = ['n', 'v']
>>> corpus = [
>>> 'He ate the sandwiches',
>>> 'Every sandwich was eaten by him'
>>> ]
>>> stemmer = PorterStemmer()
>>> print 'Stemmed:', [[stemmer.stem(token) for token in word_
tokenize(document)] for document in corpus]
>>> def lemmatize(token, tag):
>>> if tag[0].lower() in ['n', 'v']:
>>> return lemmatizer.lemmatize(token, tag[0].lower())
>>> return token
>>> lemmatizer = WordNetLemmatizer()
>>> tagged_corpus = [pos_tag(word_tokenize(document)) for document in
corpus]
>>> print 'Lemmatized:', [[lemmatize(token, tag) for token, tag in
document] for document in tagged_corpus]
Stemmed: [['He', 'ate', 'the', 'sandwich'], ['Everi', 'sandwich',
'wa', 'eaten', 'by', 'him']]
Lemmatized: [['He', 'eat', 'the', 'sandwich'], ['Every', 'sandwich',
'be', 'eat', 'by', 'him']]

#TF-IDF
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> corpus = ['The dog ate a sandwich, the wizard transfigured a
sandwich, and I ate a sandwich']
>>> vectorizer = CountVectorizer(stop_words='english') #Binary Keyword is missing 
>>> print vectorizer.fit_transform(corpus).todense()
[[2 1 3 1 1]]
{u'sandwich': 2, u'wizard': 4, u'dog': 1, u'transfigured': 3, u'ate':
0}

#Space-efficient feature vectorizing with the hashing trick
>>> from sklearn.feature_extraction.text import HashingVectorizer
>>> corpus = ['the', 'ate', 'bacon', 'cat']
>>> vectorizer = HashingVectorizer(n_features=6)
>>> print vectorizer.transform(corpus).todense()
[[-1. 0. 0. 0. 0. 0.]
[ 0. 0. 0. 1. 0. 0.]
[ 0. 0. 0. 0. -1. 0.]
[ 0. 1. 0. 0. 0. 0.]]

#Feature Generation From Images

#Let's create a feature vector for the image by reshaping its 8 x 8 matrix into a
#64-dimensional vector:
>>> from sklearn import datasets
>>> digits = datasets.load_digits()
>>> print 'Digit:', digits.target[0]
>>> print digits.images[0]
>>> print 'Feature vector:\n', digits.images[0].reshape(-1, 64)
Digit: 0
[[ 0. 0. 5. 13. 9. 1. 0. 0.]
[ 0. 0. 13. 15. 10. 15. 5. 0.]
[ 0. 3. 15. 2. 0. 11. 8. 0.]
[ 0. 4. 12. 0. 0. 8. 8. 0.]
[ 0. 5. 8. 0. 0. 9. 8. 0.]
[ 0. 4. 11. 0. 1. 12. 7. 0.]
[ 0. 2. 14. 5. 10. 12. 0. 0.]
[ 0. 0. 6. 13. 10. 0. 0. 0.]]
Feature vector:
[[ 0. 0. 5. 13. 9. 1. 0. 0. 0. 0. 13. 15. 10.
15.
5. 0. 0. 3. 15. 2. 0. 11. 8. 0. 0. 4. 12.
0.
0. 8. 8. 0. 0. 5. 8. 0. 0. 9. 8. 0. 0.
4.
11. 0. 1. 12. 7. 0. 0. 2. 14. 5. 10. 12. 0.
0.
0. 0. 6. 13. 10. 0. 0. 0.]]

#Edges and Corners
>>> import numpy as nps
>>> from skimage.feature import corner_harris, corner_peaks
>>> from skimage.color import rgb2gray
>>> import matplotlib.pyplot as plt
>>> import skimage.io as io
>>> from skimage.exposure import equalize_hist
>>> def show_corners(corners, image):
>>> fig = plt.figure()
>>> plt.gray()
>>> plt.imshow(image)
>>> y_corner, x_corner = zip(*corners)
>>> plt.plot(x_corner, y_corner, 'or')
>>> plt.xlim(0, image.shape[1])
>>> plt.ylim(image.shape[0], 0)
>>> fig.set_size_inches(np.array(fig.get_size_inches()) * 1.5)
>>> plt.show()
>>> mandrill = io.imread('/home/gavin/PycharmProjects/masteringmachine-
learning/ch4/img/mandrill.png')
>>> mandrill = equalize_hist(rgb2gray(mandrill))
>>> corners = corner_peaks(corner_harris(mandrill), min_distance=2)
>>> show_corners(corners, mandrill)

#Sift and Surf
>>> import mahotas as mh
>>> from mahotas.features import surf
>>> image = mh.imread('zipper.jpg', as_grey=True)
>>> print 'The first SURF descriptor:\n', surf.surf(image)[0]
>>> print 'Extracted %s SURF descriptors' % len(surf.surf(image))
The first SURF descriptor:
[ 6.73839947e+02 2.24033945e+03 3.18074483e+00 2.76324459e+03
-1.00000000e+00 1.61191475e+00 4.44035121e-05 3.28041690e-04
2.44845817e-04 3.86297608e-04 -1.16723672e-03 -8.81290243e-04
1.65414959e-03 1.28393061e-03 -7.45077384e-04 7.77655540e-04
1.16078772e-03 1.81434398e-03 1.81736394e-04 -3.13096961e-04
3.06559785e-04 3.43443699e-04 2.66200498e-04 -5.79522387e-04
1.17893036e-03 1.99547411e-03 -2.25938217e-01 -1.85563853e-01
2.27973631e-01 1.91510135e-01 -2.49315698e-01 1.95451021e-01
2.59719480e-01 1.98613061e-01 -7.82458546e-04 1.40287015e-03
2.86712113e-03 3.15971628e-03 4.98444730e-04 -6.93986983e-04
1.87531652e-03 2.19041521e-03 1.80681053e-01 -2.70528820e-01
2.32414943e-01 2.72932870e-01 2.65725332e-01 3.28050743e-01
2.98609869e-01 3.41623138e-01 1.58078002e-03 -4.67968721e-04
2.35704122e-03 2.26279888e-03 6.43115065e-06 1.22501486e-04
1.20064616e-04 1.76564805e-04 2.14148537e-03 8.36243899e-05
2.93382280e-03 3.10877776e-03 4.53469215e-03 -3.15254535e-04
6.92437341e-03 3.56880279e-03 -1.95228401e-04 3.73674995e-05
7.02700555e-04 5.45156362e-04]
Extracted 994 SURF descriptors


#Data Standardization
>>> from sklearn import preprocessing
>>> import numpy as np
>>> X = np.array([
>>> [0., 0., 5., 13., 9., 1.],
>>> [0., 0., 13., 15., 10., 15.],
>>> [0., 3., 15., 2., 0., 11.]
>>> ])
>>> print preprocessing.scale(X)
[[ 0. -0.70710678 -1.38873015 0.52489066 0.59299945
-1.35873244]
[ 0. -0.70710678 0.46291005 0.87481777 0.81537425
1.01904933]
[ 0. 1.41421356 0.9258201 -1.39970842 -1.4083737
0.33968311]]

#Case Study: Spam Classification using Logistic Regression

# http://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
>>> import pandas as pd
>>> df = pd.read_csv('data/SMSSpamCollection', delimiter='\t',
header=None)
>>> print df.head()
0 1
0 ham Go until jurong point, crazy.. Available only ...
1 ham Ok lar... Joking wif u oni...
2 spam Free entry in 2 a wkly comp to win FA Cup fina...
3 ham U dun say so early hor... U c already then say...
4 ham Nah I don't think he goes to usf, he lives aro...
[5 rows x 2 columns]
>>> print 'Number of spam messages:', df[df[0] == 'spam'][0].count()
>>> print 'Number of ham messages:', df[df[0] == 'ham'][0].count()
Number of spam messages: 747
Number of ham messages: 4825

>>> import numpy as np
>>> import pandas as pd
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> from sklearn.linear_model.logistic import LogisticRegression
>>> from sklearn.cross_validation import train_test_split, cross_val_
score

>>> df = pd.read_csv('data/SMSSpamCollection', delimiter='\t',
header=None)
>>> X_train_raw, X_test_raw, y_train, y_test = train_test_split(df[1],
df[0])

#TFIDF
>>> vectorizer = TfidfVectorizer()
>>> X_train = vectorizer.fit_transform(X_train_raw)
>>> X_test = vectorizer.transform(X_test_raw)

>>> classifier = LogisticRegression()
>>> classifier.fit(X_train, y_train)
>>> predictions = classifier.predict(X_test)
>>> for i, prediction in enumerate(predictions[:5]):
>>> print 'Prediction: %s. Message: %s' % (prediction, X_test_
raw[i])

#Error Metrics
>>> from sklearn.metrics import confusion_matrix
>>> import matplotlib.pyplot as plt
>>> y_test = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
>>> y_pred = [0, 1, 0, 0, 0, 0, 0, 1, 1, 1]
>>> confusion_matrix = confusion_matrix(y_test, y_pred)
>>> print(confusion_matrix)
>>> plt.matshow(confusion_matrix)
>>> plt.title('Confusion matrix')
>>> plt.colorbar()
>>> plt.ylabel('True label')
>>> plt.xlabel('Predicted label')
>>> plt.show()
[[4 1]
[2 3]]

>>> from sklearn.metrics import accuracy_score
>>> y_pred, y_true = [0, 1, 1, 0], [1, 1, 1, 1]
>>> print 'Accuracy:', accuracy_score(y_true, y_pred)
Accuracy: 0.5

#Cross Validated Scores
>>> import numpy as np
>>> import pandas as pd
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> from sklearn.linear_model.logistic import LogisticRegression
>>> from sklearn.cross_validation import train_test_split, cross_val_
score
>>> df = pd.read_csv('data/sms.csv')
>>> X_train_raw, X_test_raw, y_train, y_test = train_test_
split(df['message'], df['label'])
>>> vectorizer = TfidfVectorizer()
>>> X_train = vectorizer.fit_transform(X_train_raw)
>>> X_test = vectorizer.transform(X_test_raw)
>>> classifier = LogisticRegression()
>>> classifier.fit(X_train, y_train)
>>> scores = cross_val_score(classifier, X_train, y_train, cv=5)
>>> print np.mean(scores), scores
Accuracy 0.956217208018 [ 0.96057348 0.95334928 0.96411483
0.95454545 0.94850299]

#Precision
>>> import numpy as np
>>> import pandas as pd
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> from sklearn.linear_model.logistic import LogisticRegression
>>> from sklearn.cross_validation import train_test_split, cross_val_
score
>>> df = pd.read_csv('data/sms.csv')
>>> X_train_raw, X_test_raw, y_train, y_test = train_test_
split(df['message'], df['label'])
>>> vectorizer = TfidfVectorizer()
>>> X_train = vectorizer.fit_transform(X_train_raw)
>>> X_test = vectorizer.transform(X_test_raw)
>>> classifier = LogisticRegression()
>>> classifier.fit(X_train, y_train)
>>> precisions = cross_val_score(classifier, X_train, y_train, cv=5,
scoring='precision')
>>> print 'Precision', np.mean(precisions), precisions
>>> recalls = cross_val_score(classifier, X_train, y_train, cv=5,
scoring='recall')
>>> print 'Recalls', np.mean(recalls), recalls
Precision 0.992137651822 [ 0.98717949 0.98666667 1.
0.98684211 1. ]
Recall 0.677114261885 [ 0.7 0.67272727 0.6 0.68807339
0.72477064]

#F1 Score
>>> f1s = cross_val_score(classifier, X_train, y_train, cv=5,
scoring='f1')
>>> print 'F1', np.mean(f1s), f1s
F1 0.80261302628 [ 0.82539683 0.8 0.77348066 0.83157895


#
>>> import numpy as np
>>> import pandas as pd
>>> import matplotlib.pyplot as plt
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> from sklearn.linear_model.logistic import LogisticRegression
>>> from sklearn.cross_validation import train_test_split, cross_val_
score
>>> from sklearn.metrics import roc_curve, auc
>>> df = pd.read_csv('data/sms.csv')
>>> X_train_raw, X_test_raw, y_train, y_test = train_test_
split(df['message'], df['label'])
>>> vectorizer = TfidfVectorizer()
>>> X_train = vectorizer.fit_transform(X_train_raw)
>>> X_test = vectorizer.transform(X_test_raw)
>>> classifier = LogisticRegression()
>>> classifier.fit(X_train, y_train)
>>> predictions = classifier.predict_proba(X_test)
>>> false_positive_rate, recall, thresholds = roc_curve(y_test,
predictions[:, 1])
>>> roc_auc = auc(false_positive_rate, recall)
>>> plt.title('Receiver Operating Characteristic')
>>> plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' %
roc_auc)
>>> plt.legend(loc='lower right')
>>> plt.plot([0, 1], [0, 1], 'r--')
>>> plt.xlim([0.0, 1.0])
>>> plt.ylim([0.0, 1.0])
>>> plt.ylabel('Recall')
>>> plt.xlabel('Fall-out')
>>> plt.show()

#Grid Search
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
pipeline = Pipeline([
('vect', TfidfVectorizer(stop_words='english')),
('clf', LogisticRegression())
])
parameters = {
'vect__max_df': (0.25, 0.5, 0.75),
'vect__stop_words': ('english', None),
'vect__max_features': (2500, 5000, 10000, None),
'vect__ngram_range': ((1, 1), (1, 2)),
'vect__use_idf': (True, False),
'vect__norm': ('l1', 'l2'),
'clf__penalty': ('l1', 'l2'),
'clf__C': (0.01, 0.1, 1, 10),
}



#Should be run as script 
if __name__ == "__main__":
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1,
verbose=1, scoring='accuracy', cv=3)
df = pd.read_csv('data/sms.csv')
X, y, = df['message'], df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y)
grid_search.fit(X_train, y_train)
print 'Best score: %0.3f' % grid_search.best_score_
print 'Best parameters set:'
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
print '\t%s: %r' % (param_name, best_parameters[param_name])
predictions = grid_search.predict(X_test)
print 'Accuracy:', accuracy_score(y_test, predictions)
print 'Precision:', precision_score(y_test, predictions)
print 'Recall:', recall_score(y_test, predictions)



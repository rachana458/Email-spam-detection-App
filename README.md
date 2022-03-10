# Email-spam-detection-App-Through-HTML
Step-1:install libraries

first you need to install the required libraries pandas, sklearn, and joblib.
You can install it easily through:
pip install pandas
pip install sklearn
pip install joblib


Step-2:Import libraries

we will import libraries here. 
Here are the code
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
pandas is a python package providing fast, flexible, and expressive data structures designed to make working with "relational" or "labeled" data.
 train test split from sklearn is used to Split arrays or matrices into random train and test subsets.
The CountVectorizer from sklearn is used to convert data into machine learning language or in vector.
MultinomialNB from sklearn naive Bayes is used in text classification and detection.
joblib help in making pkl file.


Step-3: Reading the CSV file

we will rthe ead CSV file here. read CSV will help us to read our CSV file.
from df. head we will get the head of our CSV data file. 
CSV file link:
https://www.kaggle.com/uciml/sms-spam-collection-dataset
Here are the code:
df = pd.read_csv('spam.csv')
df.head(5)
df.Category.unique()


Step-4:Creating a new column

we will create a new column showing ham 0 and spam as 1.
Here are the code
df['spam'] = df['Category'].apply(lambda x: 1 if x=='spam' else 0)
df.head()


Step-5:Spliting the train and test data

we will split test and train data so, that we can train our machine and then 
test it to know the accuracy of our machine.
Here are the code:
x_train , x_test ,y_train, y_test = train_test_split(df.Message,df.spam,test_size=0.2 , random_state =42)
len(x_train)
len(x_test)


Step-6:Converting data into machine level language

As computers cannot understand human language. we make the machine understand it. so, here we are using Count Vectorizer to convert the data into machine-level language. First, we will convert x train into machine-level language. Then, we will use MultinomialNB for text detection. we will convert x test into machine-level language and check the accuracy of the x test and y test.
Here are the code:
v= CountVectorizer()
cv_messages = v.fit_transform(x_train.values)
cv_messages.toarray()[0:5]
model = MultinomialNB()
model.fit(cv_messages , y_train)
email = [
 'upto 30% discount on parking, exclusive offer just for you. Dont miss this reward!',
 'ok lar…joking wif u oni…'
]
email_count = v.transform(email)
model.predict(email_count)
x_test_count = v.transform(x_test)
model.score(x_test_count,y_test)
clf = Pipeline([
 ('vectorizer' , CountVectorizer()),
 ('nb' , MultinomialNB())
]
)
clf.fit(x_train,y_train)
email = [
 'up to 30% discount on parking, exclusive offer just for you. Don't miss this reward!',
 'ok lar…joking wif u oni…'
]
clf.predict(email)
clf.score(x_test,y_test)


Step-7:Making pkl file

we will make a pkl file from joblib.
Here are the code:
joblib.dump(clf , 'spam_model.pkl')

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#model
df=pd.read_csv("C:\\Users\\SHRIRAJ\\Desktop\\project1\\emails.csv")
df.drop_duplicates(inplace=True)
X=df['text'].values
Y=df["spam"].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2 , random_state= 0)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train=cv.fit_transform(X_train)
x_train.toarray()
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(x_train,y_train)



#streamlit code
st.title("Spam Email Classifier")
nav = st.sidebar.radio("Navigation", ["Check Email", "Statistics"])
if nav == "Check Email":
    st.header("Know whether ham or spam")
    text = st.text_input("Enter a string:")
    if text:
        clean_mail = cv.transform([text])
        check = nb.predict(clean_mail)[0]
        res = "valid email" if check == 0 else "spam email"
        if st.button("Predict"):
            st.success(f"Your email is {res}")
    else:
        st.warning("Please enter a string.")

if nav == "Statistics":
    st.header("Statistics")
    fig, ax = plt.subplots()  # Create a Matplotlib figure object
    sns.countplot(df['spam'], ax=ax)  # Plot using Matplotlib
    st.pyplot(fig)  # Pass the figure object to st.pyplot()

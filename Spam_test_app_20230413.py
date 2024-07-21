import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

st.title('垃圾信偵測器')

st.write('請輸入一段文字，我們將判斷它是否為垃圾信。')

text = st.text_input('輸入文字')

if text:
    # 載入資料集
    df = pd.read_csv('spam.csv', encoding='latin-1')
    # 切割資料集
    X_train = df['v2']
    y_train = df['v1']
    # 特徵提取
    cv = CountVectorizer()
    X_train_cv = cv.fit_transform(X_train)
    # 訓練模型
    mnb = MultinomialNB()
    mnb.fit(X_train_cv, y_train)
    # 預測結果
    text_cv = cv.transform([text])
    result = mnb.predict(text_cv)[0]
    
    if result == 'spam':
        st.write('這是一封垃圾信。')
    else:
        st.write('這不是垃圾信。')

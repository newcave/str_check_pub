import streamlit as st

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# github에서 직접 수정 후 테스트
#for i in range(1, 5):
#    print ("hello-test001")

import time
# import numpy as np #중복
 
progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
last_rows = np.random.randn(1, 1)
chart = st.line_chart(last_rows)

for i in range(1, 101):
    new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
    status_text.text("%i%% Complete" % i)
    chart.add_rows(new_rows)
    progress_bar.progress(i)
    last_rows = new_rows
    time.sleep(0.05)

progress_bar.empty()

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")

#==========================================================================
# classifier 예제

def user_input_features() :
  sepal_length = st.sidebar.slider('sepal_length',4.3, 7.9, 5.4)
  sepal_width = st.sidebar.slider('sepal_width',2.0, 4.4, 3.4)
  petal_length = st.sidebar.slider('petal_length',1.0, 6.9, 1.3)
  petal_width = st.sidebar.slider('petal_width',0.1, 2.5, 0.2)
  data = {'sepal_length' : sepal_length,
          'sepal_width' : sepal_width,
          'petal_length' : petal_length,
          'petal_width' : petal_width
          }
  features = pd.DataFrame(data, index=[0])
  return features

def main():
	#st.title("Awesome Streamlit for MLDDD")
	#st.subheader("How to run streamlit from colab")
  st.write("""
  # Simple Iris Flower Prediction WebApp

  This app predicts the **Iris flower** type!
  
  """)

  st.sidebar.header('User Input Parameters')

  df= user_input_features()

  st.subheader("파라미터를 설정해주세요.")
  st.write(df)

  iris = datasets.load_iris()
  x=iris.data
  y=iris.target

  clf = RandomForestClassifier()
  clf.fit(x,y)

  predict_ = clf.predict(df)
  predict_proba = clf.predict_proba(df)

  st.subheader("Iris 종류 ")
  st.write(iris.target_names)

  st.subheader("예측된 꽃종류")
  st.write(iris.target_names[predict_])

  st.subheader("예측된 꽃종류2")
  st.write(predict_)

  st.subheader("예측된 꽃종류3")
  st.write(iris.target_names)


  st.subheader("꽃종류별 예측 확률")
  st.write(predict_proba)

if __name__ == '__main__':
	main()

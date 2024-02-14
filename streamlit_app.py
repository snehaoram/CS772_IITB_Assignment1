# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 21:26:43 2024

@author: Sneha Oram
"""
import numpy as np
import streamlit as st
import pandas as pd
import pickle

pickle_in = open('parameters.pkl', 'rb')
param = pickle.load(pickle_in)

pickle_in1 = open('Hold_out_accuracy.pkl', 'rb')
acc1 = pickle.load(pickle_in1)

pickle_in2 = open('Whole_dataset_accuracy.pkl', 'rb')
acc = pickle.load(pickle_in2)


#from PIL import image

def predict(W1, W2, b1, b2, input):
  neth1 = np.dot(input, W1) + b1
  H1 = sigmoid(neth1)
  neto1 = np.dot(H1, W2) + b2
  O1 = sigmoid(neto1)
  p = O1 > 0.5
  return p.astype(int)


def main():
    st.title('Assignment_1')
    #st.image('Screenshot_2024-02-13_232956.png', caption='Neural Network Architecture')
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Palindrome Check App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    for i in range(4):
      st.write(i+1, " fold accuracy (#inputs = 256): ", acc1[i])

    st.write("Whole dataset accuracy (#inputs = 1024): ", acc)

    st.write("Following results are final weights and biases")
    st.write(param)
    
    num = st.text_input("Enter number: ", "Type here")
    inp = np.array([int(digit) for digit in num])
    inp = inp.reshape(1, 10)
    result = 0
    if st.button("Predict"):
        k = predict(param["W1"], param["W2"], param["b1"], param["b2"], inp)
    st.success("The input is {}".format(k))
    #if k == 1:
      #st.write('Class1, Palindrome')
    #else:
      #st.write('Class0, Not Palindrome')


if __name__ == '__main__':
    main()
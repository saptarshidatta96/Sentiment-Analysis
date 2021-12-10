import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing import text
from tensorflow import keras


st.title("Sentiment Analysis")
st.write('This is a Dummy Application which can analyse the sentiment of a Text Input.')
st.write('The Model is trained on IMDB Movie Review data by Stanford using LSTM Blocks')

@st.cache()

def sequence_vectorizer(data):
    data = preprocess_data(uploaded_data)
    tokenizer = text.Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(data)
    x_train = tokenizer.texts_to_sequences(data)
    max_length = len(max(x_train, key=len))

    if max_length > 500:
        max_length = 500

    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    print(x_train)
    return x_train, tokenizer.word_index

def preprocess_data(uploaded_data):
    data = []
    data.append(uploaded_data)
    return data
    

def load_model(weights_file):
    model = keras.models.load_model(weights_file)
    return model
    

def load_predict(uploaded_data ,weights_file):
    x_data, _ = sequence_vectorizer(uploaded_data)
    model = load_model(weights_file)
    prediction = model.predict(x_data)
    print(prediction[0][0])
    return prediction


if __name__ == "__main__":
    uploaded_data = st.text_input("label goes here", "default_value_goes_here")
    if uploaded_data is not None:
        if st.button("Predict"):
            st.write("")
            st.write("Classifying...")
            pred_score = load_predict(uploaded_data, 'sentiment_analysis_LSTM_trained_model.h5')
            st.write('The sentiment of the sentence is:')
            st.write(pred_score[0][0])
            

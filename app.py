from flask import Flask, request, render_template
import pickle
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

with open('x_tokenizer.pickle', 'rb') as handle:
    x_tokenizer = pickle.load(handle)

with open('y_tokenizer.pickle', 'rb') as handle:
    y_tokenizer = pickle.load(handle)


max_text_len=100
max_summary_len=15

reverse_target_word_index=y_tokenizer.index_word
reverse_source_word_index=x_tokenizer.index_word
target_word_index=y_tokenizer.word_index


# Encode the input sequence to get the feature vector
encoder_model = tf.keras.models.load_model("encoder_model.h5")
# Final decoder model
decoder_model = tf.keras.models.load_model("decoder_model.h5")




def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    
    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
      
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]
        
        if(sampled_token!='eostok'):
            decoded_sentence += ' '+sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'eostok'  or len(decoded_sentence.split()) >= (max_summary_len-1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():

    x_tr = [request.form['text']]

    x_tr_seq    =   x_tokenizer.texts_to_sequences(x_tr) 

    #padding zero upto maximum length
    x_tr    =   pad_sequences(x_tr_seq,  maxlen=max_text_len, padding='post')

    return 'Your Text Summery entered: {}'.format(decode_sequence(x_tr[0].reshape(1,max_text_len)))


if __name__ == '__main__':
    app.run(host='0.0.0.0')
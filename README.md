# 📊 Sentiment Analysis Using RNN & GRU AI Project
This project performs sentiment classification (Positive, Negative, Natural) on text data using Recurrent Neural Networks, specifically comparing SimpleRNN and GRU-based architectures.

✅ Final deployed model: GRU, due to better generalization and validation stability.
❌ LSTM & Bidirectional layers were tested but excluded due to overfitting or poor generalization.
## Project Overview

- **Model**: Recurrent Neural Network (RNN) / Gated Recurrent Unit (GRU)
- **Datasets**: Provided training and testing datasets
- **Goal**: To build a model that can predict the sentiment of a given text

## File Structure

- `Sentiment Analysis using RNN & GRU.ipynb`: Jupyter notebook containing the entire workflow, from data preprocessing to model evaluation.
- `train.csv`: Training dataset.
- `test.csv`: Testing dataset.

## Dataset
- Source: Pre-split CSVs (train.csv, test.csv)
- Size:
Train: 27,481 samples
Test: Remaining from dataset
## Labels:
- positive → 0
- negative → 1
- natural → 2
## Technologies Used
- Python, NumPy, Pandas
- TensorFlow / Keras
- Sequential API, GRU, Embedding
- Matplotlib (for future plots)
- Jupyter Notebook
- Pickle (for saving Tokenizer)

You can install these dependencies using the following command:

```bash
pip install pandas numpy tensorflow sklearn matplotlib
```
## Detailed Steps
1. Data Preprocessing
Load the training and testing datasets.
Perform text preprocessing such as tokenization and padding.
Encode the sentiment labels into numerical values.
3. Model Building
Define the RNN and GRU models.
Compile the models with appropriate loss function and optimizer.
Train the model on the training data.
4. Model Evaluation
Evaluate the model's performance on the testing data.
5. Predictions
Use the trained model to predict the sentiment of new text samples.
Display the predicted sentiment for the given text.
Steps to Analyze the Datasets and Create Descriptions
## Steps to Analyze the Datasets 
1. Load the Datasets:
Use pandas to load your CSV files.
```bash
import pandas as pd
train_df = pd.read_csv('train.csv', encoding='ISO-8859-1')
test_df = pd.read_csv('test.csv', encoding='ISO-8859-1')
```
2. Preprocess the Data:
Handle missing values, if any.
Tokenize text data and prepare it for the RNN model.
```bash
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(train_df['text'])

X_train = tokenizer.texts_to_sequences(train_df['text'])
X_test = tokenizer.texts_to_sequences(test_df['text'])

X_train = pad_sequences(X_train, padding='post', maxlen=35)
X_test = pad_sequences(X_test, padding='post', maxlen=35)

```
3.Build and Train the RNN and GRU Model:

Create and compile your RNN model
```bash
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

model = Sequential()
model.add(Embedding(input_dim=20000, output_dim=5, input_length=35))
model.add(SimpleRNN(32,return_sequences=False))
model.add(Dense(3,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=["accuracy"])
history = model.fit(X_train , y_train , epochs=10,validation_data=(X_test,y_test))

```
Create and compile your GRU  model
```bash
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, GRU

model2 = Sequential([
    Embedding(input_dim=20000, output_dim=4, input_length=35),
    GRU(40),
    Dense(3, activation='softmax')
])

model2.compile(optimizer='adam',loss='categorical_crossentropy',metrics=["accuracy"])
history1 = model2.fit(X_train , y_train , epochs=10,validation_data=(X_test,y_test))

```
make prediction function 
```bash
def prediction(text):
    # Prediction funtion 
    sequence = tokenizer.texts_to_sequences([text])
    padding = pad_sequences(sequence , padding='post',maxlen=35)
    pred = model2.predict(padding)
    pred_class = pred.argmax(axis=-1)
    if pred_class[0] == 0:
        print("Positive Sentiment");
    elif pred_class[0] == 1:
        print("Negative Sentiment")
    else :
        print("Natural Sentiment")
```
make prediction
```
prediction('Completely disappointed with the service.')
```
## Example Predictions
#The notebook includes examples of predicting the sentiment of new text samples. Here are a few examples:

Input: "I absolutely loved the movie I watched last night. It was so heartwarming and inspiring!"

Predicted Sentiment: Positive
Input: "I hate the traffic in this city. It's so frustrating."

Predicted Sentiment: Negative
Input: "The weather today is nice."

Predicted Sentiment: Natural
## Export & Deployment
- Model saved: sentiment_model.h5
- Tokenizer saved: tokenizer.pkl
```
# Load model
from tensorflow.keras.models import load_model
model = load_model("sentiment_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
```
You can directly use the prediction(text) function to infer sentiment from any user input in real-time.
# Model Loading & Inference
You can use the saved model and tokenizer to make predictions on new text data:
```
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = load_model("sentiment_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Prediction function
def prediction(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, padding='post', maxlen=35)
    pred = model.predict(padded)
    pred_class = pred.argmax(axis=-1)

    if pred_class[0] == 0:
        print("Positive Sentiment")
    elif pred_class[0] == 1:
        print("Negative Sentiment")
    else:
        print("Natural Sentiment")
```
## Example Outputs
```
prediction('today i feel happy')
# Output: Positive Sentiment

prediction('The meeting is scheduled for 3 PM tomorrow.')
# Output: Natural Sentiment

prediction('Completely disappointed with the service.')
# Output: Negative Sentiment
```
## Project Structure
```
├── sentiment_model.h5                          # Final GRU model
├── tokenizer.pkl                               # Tokenizer dump
├── train.csv / test.csv                        # Dataset
├── Sentiment Analysis using RNN & GRU.ipynb    # Main notebook
└── README.md                                   # Project overview
```
## Conclusion
Multiple architectures were tested — GRU outperformed SimpleRNN in validation accuracy and consistency.

LSTM and Bidirectional layers were explored but not included due to model overfitting.

This model is deployable with fast prediction (≈25ms/image), making it suitable for real-time sentiment analysis tasks.
Feel free to explore the notebook and experiment with the model on your own text data. If you have any questions or suggestions, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

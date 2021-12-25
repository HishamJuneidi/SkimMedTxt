# git clone https://github.com/Franck-Dernoncourt/pubmed-rct.git
# ls pubmed-rct

# # Check what files are in the PubMed_20K dataset 
# !ls pubmed-rct/PubMed_20k_RCT_numbers_replaced_with_at_sign

# Start by using the 20k dataset
data_dir = "pubmed-rct/PubMed_20k_RCT_numbers_replaced_with_at_sign/"

# Check all of the filenames in the target directory
import os
filenames = [data_dir + filename for filename in os.listdir(data_dir)]
filenames

# Create function to read the lines of a document
def get_lines(filename):
  """
  Reads filename (a text file) and returns the lines of text as a list.
  
  Args:
      filename: a string containing the target filepath to read.
  
  Returns:
      A list of strings with one string per line from the target filename.
      For example:
      ["this is the first line of filename",
       "this is the second line of filename",
       "..."]
  """
  with open(filename, "r") as f:
    return f.readlines()

train_lines = get_lines(data_dir+"train.txt")
train_lines[:20] # the whole first example of an abstract + a little more of the next one

def preprocess_text_with_line_numbers(filename):
  """Returns a list of dictionaries of abstract line data.

  Takes in filename, reads its contents and sorts through each line,
  extracting things like the target label, the text of the sentence,
  how many sentences are in the current abstract and what sentence number
  the target line is.

  Args:
      filename: a string of the target text file to read and extract line data
      from.

  Returns:
      A list of dictionaries each containing a line from an abstract,
      the lines label, the lines position in the abstract and the total number
      of lines in the abstract where the line is from. For example:

      [{"target": 'CONCLUSION',
        "text": The study couldn't have gone better, turns out people are kinder than you think",
        "line_number": 8,
        "total_lines": 8}]
  """
  input_lines = get_lines(filename) # get all lines from filename
  # print(input_lines[:3])
  abstract_lines = "" # create an empty abstract
  abstract_samples = [] # create an empty list of abstracts
  
  # Loop through each line in target file
  for line in input_lines:
    if line.startswith("###"): # check to see if line is an ID line
      abstract_id = line
      abstract_lines = "" # reset abstract string
    elif line.isspace(): # check to see if line is a new line
      abstract_line_split = abstract_lines.splitlines() # split abstract into separate lines

      # Iterate through each line in abstract and count them at the same time
      for abstract_line_number, abstract_line in enumerate(abstract_line_split):
        line_data = {} # create empty dict to store data from line
        target_text_split = abstract_line.split("\t") # split target label from text
        line_data["target"] = target_text_split[0] # get target label
        line_data["text"] = target_text_split[1].lower() # get target text and lower it
        line_data["line_number"] = abstract_line_number # what number line does the line appear in the abstract?
        line_data["total_lines"] = len(abstract_line_split) - 1 # how many total lines are in the abstract? (start from 0)
        abstract_samples.append(line_data) # add line data to abstract samples list
    
    else: # if the above conditions aren't fulfilled, the line contains a labelled sentence
      abstract_lines += line
  
  return abstract_samples

# Commented out IPython magic to ensure Python compatibility.
# # Get data from file and preprocess it
# 
# %%time
train_samples = preprocess_text_with_line_numbers(data_dir + "train.txt")
val_samples = preprocess_text_with_line_numbers(data_dir + "dev.txt") # dev is another name for validation set
test_samples = preprocess_text_with_line_numbers(data_dir + "test.txt")
len(train_samples), len(val_samples), len(test_samples)

import pandas as pd
train_df = pd.DataFrame(train_samples)
val_df = pd.DataFrame(val_samples)
test_df = pd.DataFrame(test_samples)
train_df.head(14)

# Distribution of labels in training data
train_df.target.value_counts()

# Convert abstract text lines into lists 
train_sentences = train_df["text"].tolist()
val_sentences = val_df["text"].tolist()
test_sentences = test_df["text"].tolist()
len(train_sentences), len(val_sentences), len(test_sentences)

# View first 10 lines of training sentences
train_sentences[:10]

# One hot encode labels
from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse=False)
train_labels_one_hot = one_hot_encoder.fit_transform(train_df["target"].to_numpy().reshape(-1, 1))
val_labels_one_hot = one_hot_encoder.transform(val_df["target"].to_numpy().reshape(-1, 1))
test_labels_one_hot = one_hot_encoder.transform(test_df["target"].to_numpy().reshape(-1, 1))

# Check what training labels look like
train_labels_one_hot

# Extract labels ("target" columns) and encode them into integers 
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_df["target"].to_numpy())
val_labels_encoded = label_encoder.transform(val_df["target"].to_numpy())
test_labels_encoded = label_encoder.transform(test_df["target"].to_numpy())
print(len(label_encoder.classes_))
# Check what training labels look like

train_labels_encoded

# Get class names and number of classes from LabelEncoder instance 
num_classes = len(label_encoder.classes_)
class_names = label_encoder.classes_
num_classes, class_names

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer






# Create a pipeline (short cut)
# model_0 = Pipeline([
#   ("tf-idf", TfidfVectorizer()),
#   ("clf", MultinomialNB())
# ])

# # Fit the pipeline to the training data
# model_0.fit(X=train_sentences, 
#             y=train_labels_encoded);


# show all the steps the pipeline can replace
#instantiate CountVectorizer() 
cv=CountVectorizer() 
 
# this steps generates word counts for the words in your docs 
word_count_vector=cv.fit_transform(train_sentences)

print(word_count_vector.shape)

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True) 
tfidf_transformer.fit(word_count_vector)


# some visualization for the data in form of DataFrame
# # print idf values 
# df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"]) 
 
# # sort ascending 
# df_idf.sort_values(by=['idf_weights'])

# recall that the train_labels_encoded shape is 
print(train_labels_encoded.shape)

# count matrix 
count_vector=cv.transform(train_sentences) 
 
# tf-idf scores 
tf_idf_vector=tfidf_transformer.transform(count_vector)
print(tf_idf_vector.shape)
feature_names = cv.get_feature_names() 
 
# #get tfidf vector for first document 
# first_document_vector=tf_idf_vector[0] 
 
# # print the scores for the TF-IDF using DataFrame
# df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"]) 
# df.sort_values(by=["tfidf"],ascending=False)

# Display the shape of the vector of TF-IDF and train encoded labels to make sure the dimentions are correct
tf_idf_vector.shape, train_labels_encoded.shape

# using classifier MultinomialNB to find out which is the property class of out categories

# initialize MultinomialNB() class
nb = MultinomialNB()

# fit the tain data
nb.fit(tf_idf_vector, train_labels_encoded)

# predict the labeled class
y_pred_class = nb.predict(tf_idf_vector)
y_pred_class.shape

# testing the validation data
count_vector_val=cv.transform(val_sentences) 
 
# tf-idf scores 
tf_idf_vector_val=tfidf_transformer.transform(count_vector_val)

y_pred_class_val = nb.predict(tf_idf_vector_val)
val_labels_encoded.shape, y_pred_class_val.shape

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# Helper function to show accuracy, precision, recall, and f1 score

def calculate_results(y_true, y_pred):
  """
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.

  Args:
      y_true: true labels in the form of a 1D array
      y_pred: predicted labels in the form of a 1D array

  Returns a dictionary of accuracy, precision, recall, f1-score.
  """
  # Calculate model accuracy
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  # Calculate model precision, recall and f1 score using "weighted average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results

# Calculate baseline results
baseline_results = calculate_results(val_labels_encoded, y_pred_class_val)
baseline_results

from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt 


cm = confusion_matrix(val_labels_encoded, y_pred_class_val)

catago =  list(label_encoder.classes_)   

ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);
ax.set_yticklabels([i for i in catago], rotation=0)
ax.set_xticklabels([i for i in catago], rotation=90)
# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix Method 1'); 
ax.xaxis.set_ticklabels(catago); 
ax.yaxis.set_ticklabels(catago);

TARGET = test_df.iloc[8][0] # BACKGROUND
TEXT = [test_df.iloc[8][1]]
print(TARGET, TEXT)

# testing the validation data
TEXT1 = ['We are aiming to improve our results.']
TEXT2 = ['Stay home from work, school and public areas.']
TEXT3 = ['continue to watch for signs and symptoms of COVID-19, especially if you’ve been around someone who is sick.']
TEXT4 = ['We gain more points because of that']
Temp_text=cv.transform(TEXT) 
 
# tf-idf scores 
tf_idf_vector_Temp_text=tfidf_transformer.transform(Temp_text)

y_pred_Temp_text = nb.predict(tf_idf_vector_Temp_text)
y_pred_Temp_text_proba = nb.predict_proba(tf_idf_vector_Temp_text)
# y_pred_Temp_text, label_encoder.classes_, y_pred_Temp_text_proba
index0 = np.argmax(y_pred_Temp_text_proba, axis = 1)[0]
catagories = list(label_encoder.classes_)
catagories[index0]
import pickle
model_filename = 'ufo-model_1.pkl'
pickle.dump(nb, open(model_filename,'wb'))

# %load_ext tensorboard
# %tensorboard --logdir logs



"""**Improve our baseline**

Start second project
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# How long is each sentence on average?
sent_lens = [len(sentence.split()) for sentence in train_sentences]
avg_sent_len = np.mean(sent_lens)
avg_sent_len # return average sentence length (in tokens)

# What's the distribution look like?
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

figure(figsize=(8,6))
plt.hist(sent_lens, bins=30);
plt.xlabel("Sentence Length")

# Maximum sentence length in the training set
max(sent_lens)

# We do not need to train on sentence with 296 length because the majority of sentence with not reach that length
# Lets find out what is the lenght for 95% of the sentence 

# How long of a sentence covers 95% of the lengths?
output_seq_len = int(np.percentile(sent_lens, 95))
output_seq_len

max_tokens = set()

txt = preprocess_text_with_line_numbers(data_dir + "train.txt")

for x in txt:
  temp = x['text'].split()
  for i in temp:
    max_tokens.add(i)
len(max_tokens)

# How many words are in our vocabulary? (taken from 3.2 in https://arxiv.org/pdf/1710.06071.pdf)
# max_tokens = 68000
max_tokens = len(max_tokens)

# Create text vectorizer
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

text_vectorizer = TextVectorization(max_tokens=max_tokens, # number of words in vocabulary
                                    output_sequence_length=55) # desired output length of vectorized sequences

# Adapt text vectorizer to training sentences
text_vectorizer.adapt(train_sentences)

# Test out text vectorizer
import random
target_sentence = random.choice(train_sentences)
print(f"Text:\n{target_sentence}")
print(f"\nTokens: {target_sentence.split()}")
print(f"\nLength of text: {len(target_sentence.split())}")
print(f"\nVectorized text:\n{text_vectorizer([target_sentence])}")

# How many words in our training vocabulary?
rct_20k_text_vocab = text_vectorizer.get_vocabulary()
print(f"Number of words in vocabulary: {len(rct_20k_text_vocab)}"), 
print(f"Most common words in the vocabulary: {rct_20k_text_vocab[:5]}")
print(f"Least common words in the vocabulary: {rct_20k_text_vocab[-5:]}")



"""From hereeeeeee"""

# Turn our data into TensorFlow Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_sentences, train_labels_one_hot))
valid_dataset = tf.data.Dataset.from_tensor_slices((val_sentences, val_labels_one_hot))
test_dataset = tf.data.Dataset.from_tensor_slices((test_sentences, test_labels_one_hot))
count = 0
for i in train_dataset:
  if count < 5:
    print(np.array(i))
  count += 1

# Take the TensorSliceDataset's and turn them into prefetched batches
train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
valid_dataset = valid_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

train_dataset

# Create token embedding layer
token_embed = layers.Embedding(input_dim=len(rct_20k_text_vocab), # length of vocabulary
                               output_dim=128, # Note: different embedding sizes result in drastically different numbers of parameters to train
                               # Use masking to handle variable sequence lengths (save space)
                               mask_zero=True,
                               name="token_embedding") 

# Show example embedding
print(f"Sentence before vectorization:\n{target_sentence}\n")
vectorized_sentence = text_vectorizer([target_sentence])
print(f"Sentence after vectorization (before embedding):\n{vectorized_sentence}\n")
embedded_sentence = token_embed(vectorized_sentence)
print(f"Sentence after embedding:\n{embedded_sentence}\n")
print(f"Embedded sentence shape: {embedded_sentence.shape}")

# Create 1D convolutional model to process sequences
inputs = layers.Input(shape=(1,), dtype=tf.string)
text_vectors = text_vectorizer(inputs) # vectorize text inputs
token_embeddings = token_embed(text_vectors) # create embedding
x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(token_embeddings)
x = layers.GlobalAveragePooling1D()(x) # condense the output of our feature vector
outputs = layers.Dense(num_classes, activation="softmax")(x)
model_1 = tf.keras.Model(inputs, outputs)

# Compile
model_1.compile(loss="categorical_crossentropy", # if your labels are integer form (not one hot) use sparse_categorical_crossentropy
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

"""Testing start

"""





"""End

"""

# Get summary of Conv1D model
model_1.summary()

# Fit the model
model_1_history = model_1.fit(train_dataset,
                              steps_per_epoch=int(0.1 * len(train_dataset)), # only fit on 10% of batches for faster training time
                              epochs=1,
                              validation_data=valid_dataset,
                              validation_steps=int(0.1 * len(valid_dataset))
                              ) 

# Make predictions (our model outputs prediction probabilities for each class)
model_1_pred_probs = model_1.predict(valid_dataset)
model_1_pred_probs

# Convert pred probs to classes
model_1_preds = tf.argmax(model_1_pred_probs, axis=1)
model_1_preds

# Calculate model_1 results
model_1_results = calculate_results(y_true=val_labels_encoded,
                                    y_pred=model_1_preds)
model_1_results

from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt 


cm = confusion_matrix(val_labels_encoded, model_1_preds)

catago =  list(label_encoder.classes_)   

ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);
ax.set_yticklabels([i for i in catago], rotation=0)
ax.set_xticklabels([i for i in catago], rotation=90)
# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix Method 2'); 
ax.xaxis.set_ticklabels(catago); 
ax.yaxis.set_ticklabels(catago);

TARGET = test_df.iloc[8][0] # BACKGROUND
TEXT = [test_df.iloc[8][1]]
print(TARGET, TEXT)

TEXT1 = ['We are aiming to improve our results.']
TEXT2 = ['Stay home from work, school and public areas.']
TEXT3 = ['continue to watch for signs and symptoms of COVID-19, especially if you’ve been around someone who is sick.']
TEXT4 = ['We gain more points because of that']

pro_arr = model_1.predict(TEXT)
index1 = np.argmax(pro_arr, axis = 1)[0]

catagories = list(label_encoder.classes_)
catagories[index1]







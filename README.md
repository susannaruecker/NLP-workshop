# Repo for the NLP-workshop

Repo containing some Jupyter Notebooks for a quick introduction to some basic NLP concepts.
The notebooks can be run in Google Colab (make sure to enable GPU!).

## Data and Task used in the examples

* in all notebooks we'll use the public datast *10kGNAD* (10k German News Articles Dataset), for more infos see https://tblock.github.io/10kGNAD/
* downloading the data is included in each notebook
* 10k **German News Articles** classified into **9 labels** (Topics/Rubrics)
    * Panorama, Web, International, Wirtschaft, Sport, Inland, Etat, Wissenschaft, Kultur 
* already split into train and test set
    * keep in mind, normally we want to have 3 splits (called train, dev/val, test), here we will use the so called test split more like you would use a development/validation split)
* so - not very surprising - we will do the task of supervised **text classification**
* note: this is a very simple NLP task, you absolutely don't need DL/BERT to solve it well, but for learning purposes it's good
* the _real_ benefit of DL-Models and BERT and friends is much more apparent in complex tasks, such as question answering, translation, sequence labelling, text generation, chatbots, summarization, ...

## some notes on goals and how to approach this:

* note: text classification is a rather simple NLP task, you absolutely don't need DL/BERT to solve it well, but for learning purposes it's good
* the _real_ benefit of DL-Models and BERT and friends is much more apparent in complex tasks, such as question answering, translation, sequence labelling, text generation, chatbots, summarization, ...
* we won't spend a lot of time in tuning the model hyperparameters (learning rate, number of layers, Dropout, ...) or think a lot about preprocessing, this holds for both the more simple models and the DL models
* so the best model performance is not the goal here
* some **goals** instead are:
   * encounter some basic NLP concepts, such as **BOW, Tokenization, Padding/Truncating, Word Embeddings, train-test-Split, pretrained Embeddings (fastText), pretrained Transformer (BERT)**
   * use them and train some simple models for the text classification task (Logistic Regression, FFN, RNN)
   * have a look at using pretrained BERT from Huggingface (Keras or PyTorch)
   * have fun with coming up with own test examples
   * maybe get some ideas what _should_ be done better... ;)

## Content Overview

* 01_BOW_sklearn_FFN.ipynb
    * data loading, small inspection, BOW features with `sklearn.CountVectorizer`, some sklearn models, classification_report, Keras-FFN, plotting learning curves, predicting unseen text
* 02_Word-Embeddings_FFN_RNN_fastText.ipynb
    * load and play around with German fastText Embeddings, use them for centroid ("bag-of-vectors") + FFN
    * then RNN with Embedding-Layer with a) no pretrained weights, b) frozen fastText weights, c) fine tuning everything
* 03a_BERT-classifier_PyTorch.ipynb
    * using pretrained `BertForSequenceClassification` from Huggingface's transformer library
    * PyTorch
* 03b_BERT-classifier_Keras.ipynb
    * trial: the same in Keras (`TFBertForSequenceClassification`)

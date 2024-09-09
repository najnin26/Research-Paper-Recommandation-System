
# Research Paper Recommandation System and Subject Area prediction using Deep Learning and LLM

This project focuses on building a machine learning-based recommendation system for research papers and predicting their subject areas using sentence embeddings from advanced natural language processing (NLP) models. The system leverages the power of sentence transformers, specifically Sentence-BERT (SBERT), to capture the semantic similarity between research papers, enabling accurate recommendations and subject area predictions.

## Methodology

1. Data Cleaning and Preprocessing: 
- Implements robust preprocessing steps to clean and prepare text data for training. This includes removing stop words, punctuation, and performing stemming to reduce words to their base form.
2. Text Vectorization: 
- Utilizes Sentence-BERT, a state-of-the-art sentence embedding model, to convert textual information into dense vectors that capture semantic meanings more effectively than traditional methods like TF-IDF or Word2Vec.
3. Model Training (MLP Classifier):
- Trains a Multilayer Perceptron (MLP) model on the vectorized data to predict the subject area of research papers. The MLP is optimized for classification tasks and offers flexibility in learning non-linear decision boundaries.
3. Recommendation System:
- Develops a recommendation system that suggests relevant research papers based on content similarity, leveraging the high-dimensional sentence embeddings generated by Sentence-BERT.
4. Model Evaluation and Deployment:
- Evaluates the model using standard classification metrics such as accuracy, precision, recall, and F1-score to ensure robust performance.
- Saves the trained model and vectorizer for future predictions, enabling easy deployment and integration into applications.
5. Model Inference and Predictions:
- Provides functionality to load the saved model and vectorizer to make predictions on new research papers, enhancing the system's usability in real-world scenarios.

## Dependencies
- Python 
- Transformers
- Tensorflow
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

## Demo

https://www.veed.io/view/0ab34b4a-6618-4a2c-86b5-180a4aa30ba3?panel=

https://www.canva.com/design/DAGQMZoXx58/3TM3p0MTq9gUpxX8BLv2ow/edit?utm_content=DAGQMZoXx58&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

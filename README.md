# Bat-Population-Decline

# README for Bat Population Decline.ipynb Notebook

## Overview
This notebook is focused on building a model to predict the likelihood of a bat population developing White-Nose Syndrome (WNS). The process involves data preprocessing, feature engineering, and training a neural network model using text-based data.

## Data Loading
The dataset is loaded from a CSV file and contains information on bat samples, including collection date, host group, location, and fungal classification.

```python
df = pd.read_csv('/content/Modeling(1).csv')
df.head()
```

## Goal
The primary goal is to predict whether WNS is present in a bat population. The model will output the certainty of its predictions.

## Data Preprocessing
1. **One-Hot Encoding**: Converts categorical variables into binary vectors.
    - `Host Group`
    - `Fungus Classification` (Phylum, Class, Order, Family, Genus, Species)

2. **Concatenation**: Combines relevant features into a single DataFrame for modeling.

```python
dummy_host = pd.get_dummies(df["Host Group"])
dummy_phylum = pd.get_dummies(df["Fungus Classification - Phylum"])
# Other dummy variables...

one_hot_df = pd.concat([df["Sample Collection Date"], dummy_host, df["CFU"], dummy_phylum, ... , df["pd_present_in_pop"]], axis=1)
```

3. **String Aggregation**: Combines specific columns into a single text string for NLP processing.

```python
df['combined'] = df[["Host Group", "CFU", "Fungus Classification - Phylum", ...]].astype(str).agg(' '.join, axis=1)
```

## Model Building
1. **Tokenization**: Uses BERT tokenizer to convert text data into numerical format.
2. **TF-IDF Vectorization**: Converts text data into TF-IDF features.

```python
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
model = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased')
X_encoded = tokenizer(text_list, padding=True, truncation=True, return_tensors='pt')["input_ids"]

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
```

3. **Neural Network Model**: Constructs and compiles a neural network using Keras.

```python
nn_model = keras.Sequential()
nn_model.add(keras.layers.InputLayer(input_shape=(vocabulary_size,)))
nn_model.add(keras.layers.Dense(units=64, activation='relu'))
nn_model.add(keras.layers.Dropout(.25))
# Other layers...
nn_model.add(keras.layers.Dense(units=1, activation='sigmoid'))

sgd_optimizer = keras.optimizers.SGD(learning_rate=0.1)
loss_fn = keras.losses.BinaryCrossentropy(from_logits=False)
nn_model.compile(optimizer=sgd_optimizer, loss=loss_fn, metrics=['accuracy'])
```

## Training
The model is trained on the training data with validation split, and training progress is logged.

```python
num_epochs = 25
history = nn_model.fit(X_train_tfidf.toarray(), y_train, epochs=num_epochs, verbose=1, validation_split=0.2)
```

## Evaluation
The model's performance is evaluated on the test data, and results are displayed.

```python
loss, accuracy = nn_model.evaluate(X_test_tfidf.toarray(), y_test)
print('Loss:', loss, 'Accuracy:', accuracy)
```

## Predictions
The model predicts the presence of WNS in the test set, and results are printed for random samples.

```python
probability_predictions = nn_model.predict(X_test_tfidf.toarray())
# Print predictions for random samples
```

## Visualization
Training and validation loss and accuracy are plotted to visualize the model's performance.

```python
plt.plot(range(1, num_epochs + 1), history.history['loss'], label='Training Loss')
plt.plot(range(1, num_epochs + 1), history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('NLP Model Loss')
plt.show()
```

## Conclusion
The Bat Population Decline.ipynb notebook demonstrates the process of building a predictive model for WNS in bat populations using a combination of data preprocessing, NLP techniques, and neural network modeling.





# README for Random_Forest_Model.ipynb Notebook

This notebook explores the same bat population decline issue as above, but uses non-NLP techniques as an approach.

## Setup Instructions

### Running the Notebook

To run this notebook, follow these steps:

1. **Open in Google Colab**: Click on the "Open in Colab" badge.
2. **Mount Google Drive**: When prompted, mount your Google Drive to access the dataset and save any outputs.
3. **Install Required Packages**: If necessary, install scikit-learn by running `pip install -U scikit-learn`.
4. **Run All Cells**: Execute all cells in the notebook sequentially.

### Requirements

- Python 3.x
- Jupyter Notebook (or Google Colab)
- Libraries: numpy, pandas, matplotlib, seaborn, scikit-learn

## Notebook Contents

### Overview

This notebook analyzes fungal CFU data collected over multiple sample collection dates.

### Data Loading and Preprocessing

- Loads data from `one_hot_model_data_without_county_and_state.csv`.
- Preprocesses the data by converting date columns to datetime objects and handling missing values in CFU counts.

### Exploratory Data Analysis (EDA)

- Visualizes the distribution of CFU counts.
- Explores correlations between CFU counts and other features.

### Modeling

- **Model Selection**: Uses various regression models such as Random Forest Regressor, Gradient Boosting Regressor, and Stacking Regressor.
- **Evaluation Metrics**: Evaluates models using Mean Squared Error (MSE) and R-squared (R2) score.

### Results

- Compares model performance using accuracy scores and visualizes prediction errors.

### Conclusion

Summarizes findings and suggests potential next steps for improving model performance.

## Conclusion

The Random Forest notebook provides a comprehensive analysis of fungal CFU prediction using machine learning techniques. For detailed insights and results, please refer to the notebook and its outputs.

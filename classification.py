#using language model to do classification of a excel file, it provide thress columns, two are  text, the other is the label, the label is the class of the text. and the label have 5 classes.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from joblib import dump, load

def train_model():
    # Load the Excel file
    df = pd.read_excel('s.xlsx')

    # Assuming the columns are named 'text1', 'text2', and 'label'
    X = df[['a', 'b']]
    y = df['c']

    # Combine the two text columns
    X['combined_text'] = X['a'] + ' ' + X['b']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X['combined_text'], y, test_size=0.2, random_state=42)

    # Create a pipeline with TF-IDF vectorizer and Random Forest classifier
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
 

    # Train the model
    pipeline.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)

  

    # Finally, save the pipeline:
    dump(pipeline, 'j.joblib')

    # Print the classification report
    #print(classification_report(y_test, y_pred))
    return pipeline

def load_model():
    return load('j.joblib') 

# Function to classify new text
def classify_text(text1, text2):
    combined_text = text1 + ' ' + text2
    pipeline = load_model()
    prediction = pipeline.predict([combined_text])
    return prediction[0]

# Example usage
def test():
    new_text1 = "Supplier Name: SAY ABOUT LIMITED , PO no.: PO-10038160	"
    new_text2 = "SAY ABOUT LIMITED  "
    pipeline = load_model()
    predicted_class = classify_text(new_text1, new_text2)
    print(f"Predicted class: {predicted_class}")

#wrap the classify_text function into a flask api
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.get('/')
def home():
    print(request.values["text1"])
    return "hello"

 

@app.route('/classify', methods=[ 'GET'])
def classify():
 
    text1 = request.values["text1"]
    text2 = request.values["text2"]
    
    if not text1 or not text2:
        return jsonify({'error': 'Both text1 and text2 are required'}), 400

    
    prediction = classify_text(text1, text2)
    print(f"Predicted class: {prediction}")
    return  prediction 

if __name__ == '__main__':
    app.run(debug=True)

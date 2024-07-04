import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import streamlit as st
from lime.lime_text import LimeTextExplainer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the classifier model
class Classifier(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(input_dim, num_labels)
        )

    def forward(self, x):
        return self.classifier(x)

# Load the model
model = Classifier(input_dim=384, num_labels=6)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model state
model_path = 'Trained-Model.pth'  # This should be the correct file name
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    model_loaded = True
except FileNotFoundError:
    st.error(f"Model file not found at path: {model_path}. Please ensure the model file is in the correct location.")
    model_loaded = False

# Load the SBERT model for embedding
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def predict_text_authenticity(text):
    # Generate embeddings
    embeddings = sbert_model.encode([text], convert_to_numpy=True)
    embeddings = torch.tensor(embeddings).float().to(device)

    # Predict with the classifier
    with torch.no_grad():
        outputs = model(embeddings)
        predicted = outputs.argmax(dim=1).item()

    # Map predicted label index to label name
    label_mapping = {
        0: 'true', 1: 'pants-fire', 2: 'false',
        3: 'barely-true', 4: 'half-true', 5: 'mostly-true'
    }
    return label_mapping[predicted]

# Function to render different styles based on prediction
def render_prediction(prediction):
    styles = {
        'true': {"background-color": "#d4edda", "color": "#155724"},  # Light green background, dark green text
        'pants-fire': {"background-color": "#ff0000", "color": "#ffffff"},  # Red background, white text
        'false': {"background-color": "#ffcccc", "color": "#000000"},  # Light red background, black text
        'barely-true': {"background-color": "#fff3cd", "color": "#856404"},  # Light yellow background, dark yellow text
        'half-true': {"background-color": "#ccffcc", "color": "#000000"},  # Light green background, black text
        'mostly-true': {"background-color": "#006400", "color": "#ffffff"},  # Dark green background, white text
    }
    style = styles[prediction]
    st.markdown(
        f"""
        <style>
        .container {{
            max-width: 800px;
            margin: auto;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        h1 {{
            color: #ffffff;  /* White text */
            font-size: 36px;
        }}
    
        p {{
            color: #555555;
            font-size: 16px;
            text-align: center;
        }}
        .prediction-box {{
            background-color: {style["background-color"]};
            color: {style["color"]};
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            animation: fadeIn 1s ease-in-out;
            font-size: 24px;
            font-weight: bold;
        }}
        @keyframes fadeIn {{
            0% {{opacity: 0;}}
            100% {{opacity: 1;}}
        }}
        .explanation {{
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            background-color: transparent;
        }}
        .explanation h2 {{
            color: #d3d3d3;  /* Greyish white text */
            font-size: 20px;
            text-align: left;
        }}
        .explanation p {{
            color: #d3d3d3; /* Greyish white text */
            font-size: 16px;
            margin: 5px 0;
            text-align: left;
        }}
        .explanation table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
            background-color: transparent;
        }}
        .explanation th, .explanation td {{
            border: 1px solid #dddddd;
            text-align: left;
            padding: 8px;
            background-color: #333333;
            color: #ffffff;
        }}
        .explanation th {{
            background-color: #444444;
        }}
        .custom-text-area label {{
            font-size: 20px;  /* Set to 20px for medium size */
        }}
        .stButton>button {{
            background-color: #3b5998; /* Facebook blue color */
            color: white;  /* Ensure the text is white */
            font-size: 20px;  /* Adjust font size if necessary */
            border-radius: 5px;
            padding: 10px;
            border: none;  /* Remove default border */
            outline: none;  /* Remove default outline */
        }}
        .stButton>button:focus {{
            border: none;  /* Remove focus border */
            outline: none;  /* Remove focus outline */
        }}
        </style>
        <div class="prediction-box">
            The sentence is predicted to be: <b>{prediction}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

# Function to explain prediction using LIME
def explain_prediction(text):
    explainer = LimeTextExplainer(class_names=['true', 'pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true'])
    
    def classifier_fn(texts):
        embeddings = sbert_model.encode(texts, convert_to_numpy=True)
        embeddings = torch.tensor(embeddings).float().to(device)
        with torch.no_grad():
            outputs = model(embeddings)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        return probabilities.cpu().numpy()

    explanation = explainer.explain_instance(text, classifier_fn, num_features=10)
    return explanation

def main():
    # Custom CSS to change font size and button color
    st.markdown("""
        <style>
        .custom-text-area label {
            font-size: 20px;  /* Set to 20px for medium size */
        }
        .stButton>button {
            background-color: #3b5998; /* Facebook blue color */
            color: white;  /* Ensure the text is white */
            font-size: 20px;  /* Adjust font size if necessary */
            border-radius: 5px;
            padding: 10px;
            border: none;  /* Remove default border */
            outline: none;  /* Remove default outline */
        }
        .stButton>button:focus {
            border: none;  /* Remove focus border */
            outline: none;  /* Remove focus outline */
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="container">
            <h1>Real-Time Fake News Detection</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    user_input = st.text_area("Enter a sentence to predict its truthfulness:", key="custom-text-area")
    
    if st.button("Predict"):
        if not model_loaded:
            st.error("Model is not loaded. Please check the model path.")
        else:
            prediction = predict_text_authenticity(user_input)
            render_prediction(prediction)
            explanation = explain_prediction(user_input)

            # Prepare explanation data for the table
            explanation_data = explanation.as_list()
            explanation_df = pd.DataFrame(explanation_data, columns=["Word", "Contribution"])

            # Display the explanation table
            table_html = """
                <div class="explanation">
                    <h2>Prediction Explanation:</h2>
                    <p>The following words contributed to the prediction:</p>
                    <table>
                        <thead>
                            <tr>
                                <th>Word</th>
                                <th>Contribution</th>
                            </tr>
                        </thead>
                        <tbody>
            """
            for word, score in explanation_data:
                table_html += f"<tr><td>{word}</td><td>{score:.4f}</td></tr>"
            table_html += """
                        </tbody>
                    </table>
                </div>
            """
            st.markdown(table_html, unsafe_allow_html=True)

            # Explanation summary
            st.markdown(
                """
                <div class="explanation">
                    <p>The table above lists the words from the input sentence that had the most significant impact on the prediction. Positive values indicate words that pushed the prediction towards being true, while negative values indicate words that pushed it towards being false.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Display the explanation graph
            st.pyplot(explanation.as_pyplot_figure())

if __name__ == "__main__":
    main()

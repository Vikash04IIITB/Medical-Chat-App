from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import random
import json
import torch
import pandas as pd
import numpy as np
from flask_cors import CORS
import warnings
import pickle
import nltk
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "http://localhost:3000",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept"]
    }
})

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    print("Error: GOOGLE_API_KEY not found in .env file")
    exit(1)

# Configure Google Generative AI
try:
    genai.configure(api_key=google_api_key)
except Exception as e:
    print(f"Error configuring Google Generative AI: {e}")
    exit(1)

# Load intents
try:
    with open('intents.json', 'r') as json_data:
        intents = json.load(json_data)
except FileNotFoundError:
    print("Error: intents.json not found")
    exit(1)
except json.JSONDecodeError:
    print("Error: Invalid JSON in intents.json")
    exit(1)

# Set environment variables for PyTorch
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Load neural net model for chat
device = torch.device('cpu')
try:
    data = torch.load("data.pth", map_location=device)
    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()
except FileNotFoundError:
    print("Error: data.pth not found")
    exit(1)
except Exception as e:
    print(f"Error loading neural net model: {e}")
    exit(1)

# Load disease prediction model
try:
    with open('ExtraTrees', 'rb') as f:
        disease_model = pickle.load(f)
except FileNotFoundError:
    print("Error: ExtraTrees model file not found")
    exit(1)
except Exception as e:
    print(f"Error loading ExtraTrees model: {e}")
    exit(1)

# Define diseases and symptoms
diseases = ['(vertigo) Paroymsal Positional Vertigo', 'AIDS', 'Acne', 'Alcoholic hepatitis', 'Allergy', 'Arthritis', 'Bronchial Asthma', 'Cervical spondylosis', 'Chicken pox', 'Chronic cholestasis', 'Common Cold', 'Dengue', 'Diabetes', 'Dimorphic hemmorhoids(piles)', 'Drug Reaction', 'Fungal infection', 'GERD', 'Gastroenteritis', 'Heart attack', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Hypertension', 'Hyperthyroidism', 'Hypoglycemia', 'Hypothyroidism', 'Impetigo', 'Jaundice', 'Malaria', 'Migraine', 'Osteoarthristis', 'Paralysis (brain hemorrhage)', 'Peptic ulcer diseae', 'Pneumonia', 'Psoriasis', 'Tuberculosis', 'Typhoid', 'Urinary tract infection', 'Varicose veins', 'hepatitis A']

symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze']

# Load symptom description and precaution CSVs
try:
    desc = pd.read_csv("data/symptom_Description.csv")
    prec = pd.read_csv("data/symptom_precaution.csv")
except FileNotFoundError:
    print("Error: symptom_Description.csv or symptom_precaution.csv not found")
    exit(1)

# Chat function for intent-based responses
def chat(sentence):
    bot_name = "Aarogya"
    try:
        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent['tag']:
                    if tag == "disease":
                        response = user_input(sentence)
                        return response["output_text"]
                    else:
                        return random.choice(intent['responses'])
        else:
            return "I do not understand..."
    except Exception as e:
        print(f"Error in chat function: {e}")
        return "Error processing your request. Please try again."

# PDF processing functions
def get_pdf_text(pdf_docs):
    try:
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""

def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        return text_splitter.split_text(text)
    except Exception as e:
        print(f"Error splitting text: {e}")
        return []

def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        print(f"Error creating vector store: {e}")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, say, "Answer is not available in the context." Do not provide incorrect information.

    Context: {context}
    Question: {question}

    Answer:
    """
    try:
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        return load_qa_chain(model, chain_type="stuff", prompt=prompt)
    except Exception as e:
        print(f"Error creating conversational chain: {e}")
        return None

def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = new_db.as_retriever(search_kwargs={"k": 20})
        docs = retriever.get_relevant_documents(user_question)
        
        chain = get_conversational_chain()
        if not chain:
            return {"output_text": "Error initializing conversational chain."}

        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response
    except Exception as e:
        print(f"Error in user_input: {e}")
        return {"output_text": "Error processing your query. Please try again."}

# API Routes
@app.route('/disease', methods=["GET"])
def home():
    return jsonify({"message": "Welcome to Aarogya Disease Prediction API"})

@app.route("/get", methods=["POST"])
def get_response():
    try:
        data = request.get_json()
        if not data or "msg" not in data:
            return jsonify({"error": "No message provided"}), 400
        user_question = data["msg"]
        response = chat(user_question)
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error in /get endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No symptoms provided"}), 400

        # Initialize features array with 218 (model's expected size)
        expected_features = 218
        features = [0] * expected_features

        # Map received symptoms to the correct indices
        mapped_symptoms = 0
        for symptom in data:
            if symptom in symptoms:
                index = symptoms.index(symptom)
                if index < expected_features:
                    features[index] = 1
                    mapped_symptoms += 1
        print(f"Mapped {mapped_symptoms} symptoms to {expected_features} features array")

        # Validate feature length
        if len(features) != expected_features:
            raise ValueError(f"Feature array length {len(features)} does not match expected {expected_features}")

        # Predict using the model
        proba = disease_model.predict_proba([features])[0]
        top5_idx = np.argsort(proba)[-5:][::-1]
        top5_proba = proba[top5_idx]
        top5_diseases = [diseases[i] for i in top5_idx]

        response = []
        for i in range(5):
            disease = top5_diseases[i]
            probability = top5_proba[i]
            disp = desc[desc['Disease'] == disease].iloc[0, 1] if disease in desc["Disease"].values else "No description available"
            precautions = prec.loc[prec['Disease'] == disease, prec.columns[1:]].values.flatten()
            precautions = [p for p in precautions if pd.notna(p) and p != '']

            response.append({
                'disease': disease,
                'probability': float(probability),
                'description': disp,
                'precautions': precautions
            })

        return jsonify(response)
    except ValueError as ve:
        print(f"Value Error in /predict endpoint: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        print(f"Error in /predict endpoint: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
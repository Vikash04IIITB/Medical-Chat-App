import React, { useState, useEffect } from 'react';
import axios from 'axios';
import ChatUI from './components/ChatUI';
import DiseasePredictor from './components/DiseasePredictor';
import './components/app.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import { Alert, Spinner } from 'react-bootstrap';
// import logger from './logger';

function App() {
  const [chatMessages, setChatMessages] = useState([]);
  const [symptoms, setSymptoms] = useState([]);
  const [diagnosisResults, setDiagnosisResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Backend URL from environment variable (or fallback to localhost)
  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:5001';

  const symptomsList = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills',
    'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting',
    'burning_micturition', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings',
    'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough',
    'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache',
    'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain',
    'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes',
    'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise',
    'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure',
    'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate',
    'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus',
    'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels',
    'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger',
    'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain',
    'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements',
    'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort',
    'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression',
    'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
    'abnormal_menstruation', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history',
    'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
    'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding',
    'distention_of_abdomen', 'history_of_alcohol_consumption', 'blood_in_sputum', 'prominent_veins_on_calf',
    'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
    'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose',
    'yellow_crust_ooze'
  ];

  // Clear selected symptoms
  const clearSymptoms = () => {
    setSymptoms([]);
    setDiagnosisResults([]);
    setError(null);
  };

  useEffect(() => {
    // logger.info('App component mounted');
    return () => {
      // logger.info('App component unmounted');
    };
  }, []);

  // Handle user chat message
  const handleUserMessage = async (message) => {
    if (!message.trim()) {
      setError('Please enter a valid message.');
      return;
    }

    setLoading(true);
    setError(null);
    const newMessage = { text: message, sender: 'user' };
    setChatMessages((prev) => [...prev, newMessage]);

    try {
      const response = await axios.post(`${BACKEND_URL}/get`, { msg: message }, {
        headers: { 'Content-Type': 'application/json' },
        withCredentials: false
      });
      const botMessage = { text: response.data.response, sender: 'bot' };
      setChatMessages((prev) => [...prev, botMessage]);
      // logger.info('Bot response received:', response.data.response);
    } catch (error) {
      const errorMsg = error.response?.data?.error || 'Failed to get bot response. Please try again.';
      setError(errorMsg);
      console.error('Error getting bot response:', error);
      // logger.error('Error getting bot response:', error);
    } finally {
      setLoading(false);
    }
  };

  // Add symptom to list
  const handleAddSymptom = (symptom) => {
    if (!symptom || symptoms.includes(symptom)) return;
    setSymptoms((prev) => [...prev, symptom]);
    setError(null);
  };

  // Predict disease based on symptoms
  const handlePredictDisease = async () => {
    if (symptoms.length === 0) {
      setError('Please select at least one symptom.');
      return;
    }

    setLoading(true);
    setError(null);
    setDiagnosisResults([]);

    try {
      const response = await axios.post(`${BACKEND_URL}/predict`, symptoms, {
        headers: { 'Content-Type': 'application/json' },
        withCredentials: false
      });
      setDiagnosisResults(response.data);
      // logger.info('Disease prediction response:', response.data);
    } catch (error) {
      const errorMsg = error.response?.data?.error || 'Failed to predict disease. Please try again.';
      setError(errorMsg);
      console.error('Error predicting disease:', error);
      // logger.error('Error predicting disease:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container-fluid h-100">
      {loading && (
        <div className="text-center my-3">
          <Spinner animation="border" variant="primary" />
        </div>
      )}
      {error && (
        <Alert variant="danger" onClose={() => setError(null)} dismissible>
          {error}
        </Alert>
      )}
      <div className="d-flex flex-row justify-content-between align-items-stretch h-100">
        <div className="col-md-8 col-xl-6 chat m-3">
          <ChatUI chatMessages={chatMessages} onUserMessage={handleUserMessage} />
        </div>
        <div className="col-xl-5 m-3">
          <DiseasePredictor
            symptoms={symptoms}
            onAddSymptom={handleAddSymptom}
            onPredictDisease={handlePredictDisease}
            diagnosisResults={diagnosisResults}
            symptomsList={symptomsList}
            clearSymptoms={clearSymptoms}
          />
        </div>
      </div>
    </div>
  );
}

export default App;
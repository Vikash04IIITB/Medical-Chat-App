import React, { useState } from 'react';
import DiagnosisResult from './DiagnosisResult';
import SymptomInput from './SymptomInput';
import 'bootstrap/dist/css/bootstrap.min.css';

const DiseasePredictor = ({
  symptoms,
  onAddSymptom,
  onPredictDisease,
  diagnosisResults,
  symptomsList,
  clearSymptoms, // Accept the clearSymptoms prop
}) => {
  const [isRefreshing, setIsRefreshing] = useState(false);

  const handleRefresh = () => {
    setIsRefreshing(true);
    onPredictDisease([]);
    onAddSymptom([]);
    clearSymptoms(); // Call the clearSymptoms prop
    setIsRefreshing(false);
  };

  const handlePredictDisease = () => {
    if (symptoms.length === 0) {
      alert('Please add at least one symptom.');
      return;
    }

    onPredictDisease(symptoms);
  };

  return (
    <div id="predictionContent" className="prediction-content">
      <h3>Medical Diagnosis</h3>
      <div>
        <div className="d-flex align-items-start mb-3">
          <SymptomInput
            symptoms={symptoms}
            onAddSymptom={onAddSymptom}
            symptomsList={symptomsList}
          />
          <div className="d-flex align-items-center ">
            <button
              id="predict"
              className="predict-btn mr-3 ml-3"
              onClick={handlePredictDisease}
              disabled={isRefreshing}
            >
              {isRefreshing ? 'Refreshing...' : 'Predict'}
            </button>
            <button
              id="refresh"
              className="refresh-btn mr-3"
              onClick={handleRefresh}
              disabled={isRefreshing}
            >
              {isRefreshing ? 'Refreshing...' : 'Refresh'}
            </button>
          </div>
        </div>
        <ul id="symptoms" className="symptoms-list ml-3">
          {symptoms.map((symptom, index) => (
            <li key={index} className="symptom-item">
              {symptom}
            </li>
          ))}
        </ul>
      </div>
      <div id="result">
        <DiagnosisResult diagnosisResults={diagnosisResults} />
      </div>
    </div>
  );
};

export default DiseasePredictor;
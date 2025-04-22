import React from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';

const DiagnosisResult = ({ diagnosisResults }) => {
  return (
    <div>
      <h2>Diagnosis Results</h2>
      {diagnosisResults.length > 0 ? (
        diagnosisResults.map((result, index) => (
          <div key={index}>
            <h3>{result.disease}</h3>
            <p>Probability: {result.probability}</p>
            <p>Description: {result.description}</p>
            <p>Precautions:</p>
            <ul>
              {result.precautions.map((precaution, index) => (
                <li key={index}>{precaution}</li>
              ))}
            </ul>
          </div>
        ))
      ) : (
        <p>No diagnosis results available.</p>
      )}
    </div>
  );
};

export default DiagnosisResult;
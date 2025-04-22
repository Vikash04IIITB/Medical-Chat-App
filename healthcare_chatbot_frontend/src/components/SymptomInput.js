// import React, { useState } from 'react';
// import 'bootstrap/dist/css/bootstrap.min.css';

// const SymptomInput = ({ onAddSymptom, onPredictDisease }) => {
//   const [symptom, setSymptom] = useState('');

//   const handleAddSymptom = () => {
//     if (symptom.trim()) {
//       onAddSymptom(symptom.trim());
//       setSymptom('');
//     }
//   };

//   return (
//     <div>
//       <input
//         type="text"
//         value={symptom}
//         onChange={(e) => setSymptom(e.target.value)}
//         placeholder="Enter a symptom"
//       />
//       <button onClick={handleAddSymptom}>Add Symptom</button>
//       <button onClick={onPredictDisease}>Predict Disease</button>
//     </div>
//   );
// };

// export default SymptomInput;


// SymptomInput.js
import React, { useState, useEffect, useRef } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';

const SymptomInput = ({ symptoms, onAddSymptom, symptomsList }) => {
  const [symptom, setSymptom] = useState('');
  const [filteredSymptoms, setFilteredSymptoms] = useState([]);
  const suggestionRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (suggestionRef.current && !suggestionRef.current.contains(event.target)) {
        setFilteredSymptoms([]);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const handleAddSymptom = (symptom) => {
    if (symptom.trim()) {
      onAddSymptom(symptom.trim());
      setSymptom('');
      setFilteredSymptoms([]);
    }
  };

  const handleSymptomChange = (e) => {
    const value = e.target.value;
    setSymptom(value);
    const filtered = symptomsList.filter((symptom) =>
      symptom.toLowerCase().startsWith(value.toLowerCase())
    );
    setFilteredSymptoms(filtered);
  };

  const handleSuggestionClick = (suggestion) => {
    handleAddSymptom(suggestion);
  };

  return (
    <div className="symptom-input-container">
      <input
        type="text"
        value={symptom}
        onChange={handleSymptomChange}
        className="symptom-input"
        placeholder="Enter a symptom"
        onKeyPress={(e) => {
          if (e.key === 'Enter') {
            handleAddSymptom(e.target.value.trim());
          }
        }}
      />
      <button id="add" className="add-symptom-btn" onClick={() => handleAddSymptom(symptom)}>
        Add Symptom
      </button>
      {filteredSymptoms.length > 0 && (
        <ul ref={suggestionRef} style={{ listStyle: 'none', padding: 0 }}>
          {filteredSymptoms.map((suggestion, index) => (
            <li
              key={index}
              onClick={() => handleSuggestionClick(suggestion)}
              style={{ cursor: 'pointer' }}
            >
              {suggestion}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default SymptomInput;
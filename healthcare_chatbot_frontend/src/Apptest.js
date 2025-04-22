// App.test.js
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import axios from 'axios';
import App from './App';

jest.mock('axios');

describe('App', () => {
  beforeEach(() => {
    axios.post.mockResolvedValue({ data: { response: 'Hello, this is a test response.' } });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  test('renders chat UI and handles user message', async () => {
    render(<App />);
    const inputBox = screen.getByPlaceholderText('Type your message here');
    fireEvent.change(inputBox, { target: { value: 'Hello' } });
    fireEvent.keyDown(inputBox, { key: 'Enter', keyCode: 13 });

    await waitFor(() => {
      const messages = screen.getAllByTestId('message');
      expect(messages.length).toBe(2);
      expect(messages[0].textContent).toBe('Hello');
      expect(messages[1].textContent).toBe('Hello, this is a test response.');
    });
  });

  test('handles adding symptoms and predicting disease', async () => {
    render(<App />);
    const symptoms = ['fever', 'cough'];
    const diagnosisResults = ['Flu', 'Common Cold'];

    for (const symptom of symptoms) {
      fireEvent.click(screen.getByText(symptom));
    }

    fireEvent.click(screen.getByText('Predict Disease'));

    await waitFor(() => {
      const resultItems = screen.getAllByTestId('diagnosis-result');
      expect(resultItems.length).toBe(diagnosisResults.length);
      resultItems.forEach((item, index) => {
        expect(item.textContent).toBe(diagnosisResults[index]);
      });
    });
  });
});
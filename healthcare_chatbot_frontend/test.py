import unittest
from unittest.mock import Mock, patch
import React from 'react';
import { render, fireEvent, waitFor, screen } from '@testing-library/react';
import axios from 'axios';
import App from './App';

class TestApp(unittest.TestCase):
    def setUp(self):
        self.mock_axios = Mock()
        self.mock_axios.post.return_value = Mock(data={'response': 'Hello, this is a test response.'})

    @patch('axios.post', side_effect=lambda *args, **kwargs: self.mock_axios.post(*args, **kwargs
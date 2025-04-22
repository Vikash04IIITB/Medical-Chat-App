import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier

class TestTrainModel(unittest.TestCase):
    def setUp(self):
        self.data = pd.read_csv('/var/lib/jenkins/workspace/SPE_finalProject/training/patients_data.csv')
        self.X = self.data.iloc[:, 1:].values
        self.y = self.data['Disease'].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)

    def test_data_shape(self):
        self.assertGreater(self.X.shape[0], 0, "No data loaded")
        self.assertGreater(self.X.shape[1], 0, "No features in data")

    def test_label_encoder(self):
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(self.y_train)
        self.assertEqual(len(np.unique(y_train_encoded)), len(le.classes_), "Encoding issue")

    def test_model_training(self):
        model = ExtraTreesClassifier()
        model.fit(self.X_train, self.y_train)
        self.assertIsNotNone(model, "Model not trained")

    def test_model_prediction(self):
        model = ExtraTreesClassifier()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        self.assertEqual(len(y_pred), self.X_test.shape[0], "Prediction issue")

if __name__ == '__main__':
    unittest.main()
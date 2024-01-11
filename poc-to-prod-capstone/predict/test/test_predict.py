import unittest
from predict.run import TextPredictionModel

class TestTextPredictionModel(unittest.TestCase):
    def test_predict(self):
        model = TextPredictionModel.from_artefacts("C:/POC/poc-to-prod-capstone/train/data/artefacts/2024-01-09-21-10-14")
        predictions = model.predict(['text to analyze'])
        self.assertIsNotNone(predictions)

if __name__ == '__main__':
    unittest.main()

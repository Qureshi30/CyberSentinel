# Test Data Directory

This directory is used for storing:
1. CSV files uploaded through the web interface
2. Prediction output files

## File Usage:

- **sample_input.csv** - CSV files uploaded via the web app are saved here temporarily
- Files with **_predictions.csv** suffix contain the original data plus prediction results

## CSV Format for Testing:

Your test CSV should have the same features (columns) as the training data, but does NOT need to include a label column since the model will predict it.

Example columns might include:
- duration
- protocol_type
- service
- flag
- src_bytes
- dst_bytes
- and other network traffic features...

The web application will automatically handle the prediction and display results.

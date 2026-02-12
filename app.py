"""
Flask Web Application
Simple web interface for network traffic classification
"""

import os
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from src.predict import predict_from_csv

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = os.path.join('data', 'test')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def index():
    """
    Render the main page
    """
    return render_template('index.html', predictions=None, error=None)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle CSV upload and make predictions
    """
    error = None
    predictions = None
    
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            error = "No file uploaded"
            return render_template('index.html', predictions=None, error=error)
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            error = "No file selected"
            return render_template('index.html', predictions=None, error=error)
        
        # Check if file is CSV
        if not file.filename.endswith('.csv'):
            error = "Please upload a CSV file"
            return render_template('index.html', predictions=None, error=error)
        
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'sample_input.csv')
        file.save(file_path)
        
        # Make predictions
        result_df = predict_from_csv(file_path)
        
        # Prepare predictions for display (limit to first 100 rows for display)
        display_limit = 100
        predictions = []
        
        for idx, row in result_df.head(display_limit).iterrows():
            pred_result = {
                'row_number': idx + 1,
                'prediction': row['Prediction'],
                'features': {}
            }
            
            # Add first few features for context (exclude the prediction column)
            feature_count = 0
            for col in result_df.columns:
                if col != 'Prediction' and feature_count < 5:
                    pred_result['features'][col] = row[col]
                    feature_count += 1
            
            predictions.append(pred_result)
        
        # Calculate summary statistics
        total = len(result_df)
        threat_count = (result_df['Prediction'] == 'Threat').sum()
        harmless_count = (result_df['Prediction'] == 'Harmless').sum()
        
        summary = {
            'total': total,
            'threat': threat_count,
            'harmless': harmless_count,
            'threat_percentage': f"{(threat_count/total*100):.1f}",
            'harmless_percentage': f"{(harmless_count/total*100):.1f}",
            'showing': min(display_limit, total)
        }
        
        return render_template('index.html', predictions=predictions, summary=summary, error=None)
    
    except FileNotFoundError as e:
        error = "Model not found. Please train the model first by running: python src/train_model.py"
        return render_template('index.html', predictions=None, error=error)
    
    except Exception as e:
        error = f"Error processing file: {str(e)}"
        return render_template('index.html', predictions=None, error=error)


if __name__ == '__main__':
    # Check if model exists
    model_path = os.path.join('model', 'intrusion_model.pkl')
    if not os.path.exists(model_path):
        print("\n" + "="*60)
        print("WARNING: Model not found!")
        print("="*60)
        print("Please train the model first by running:")
        print("  python src/train_model.py")
        print("="*60 + "\n")
    
    # Run the Flask app
    print("\n" + "="*60)
    print("CYBERSENTINEL - Network Traffic Classifier")
    print("="*60)
    print("Starting web server...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    print("Press CTRL+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

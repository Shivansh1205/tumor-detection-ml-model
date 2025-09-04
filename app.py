from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the pickled model
model = None
try:
    with open('logistic_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: Model file 'logistic_regression_model.pkl' not found!")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for tumor prediction"""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please ensure logistic_regression_model.pkl exists.'
            }), 500
        
        # Get data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract features
        tumor_size = float(data.get('tumor_size', 0))
        tumor_texture = float(data.get('tumor_texture', 0))
        
        # Validate inputs
        if tumor_size < 0 or tumor_texture < 0:
            return jsonify({'error': 'Tumor size and texture must be positive values'}), 400
        
        # Create feature array
        features = np.array([[tumor_size, tumor_texture]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        # Format response
        result = {
            'prediction': int(prediction),
            'prediction_label': 'Malignant' if prediction == 1 else 'Benign',
            'probability_benign': float(probability[0]),
            'probability_malignant': float(probability[1]),
            'confidence': float(max(probability)),
            'tumor_size': tumor_size,
            'tumor_texture': tumor_texture
        }
        
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({'error': 'Invalid input values. Please provide numeric values.'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get model parameters if available
        info = {
            'model_type': type(model).__name__,
            'features': ['Tumor Size', 'Tumor Texture'],
            'classes': model.classes_.tolist() if hasattr(model, 'classes_') else [0, 1],
            'n_features': model.n_features_in_ if hasattr(model, 'n_features_in_') else 2
        }
        
        if hasattr(model, 'coef_'):
            info['coefficients'] = model.coef_.tolist()
        if hasattr(model, 'intercept_'):
            info['intercept'] = model.intercept_.tolist()
            
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': f'Error getting model info: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    print("Starting Tumor Detection API Server...")
    print("Endpoints:")
    print("  GET  / - Main page")
    print("  POST /api/predict - Make predictions")
    print("  GET  /api/health - Health check")
    print("  GET  /api/model-info - Model information")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
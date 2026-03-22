from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from datetime import datetime, timedelta
import os
import sys

# =========================================================
# 1. ABSOLUTE PATH DEFINITIONS (CRITICAL: CHANGE THESE!)
# =========================================================

# --- Flask Directory Paths ---
# Path to the folder containing index.html (NOT the file itself!)
TEMPLATE_DIR = r'c:/Users/Sun/OneDrive/Documents/mini project cse/crypto-volatility-forecast/notebooks/templates'
# Path to the static folder (for CSS, JS, images)
STATIC_DIR = r'c:/Users/Sun/OneDrive/Documents/mini project cse/crypto-volatility-forecast/notebooks/static'

# --- Model File Paths ---
# Path to the saved Random Forest model file (rf_model.joblib)
RF_MODEL_PATH = r'c:/Users/Sun/OneDrive/Documents/mini project cse/crypto-volatility-forecast/notebooks/models/rf_model.joblib'
# Path to the saved XGBoost model file (xgb_model.joblib)
XGB_MODEL_PATH = r'C:/Users/Sun/OneDrive/Documents/mini project cse/crypto-volatility-forecast/notebooks/models/xgb_model.joblib'
# Path to the saved Scaler file (scaler.joblib)
SCALER_PATH = r'C:/Users/Sun/OneDrive/Documents/mini project cse/crypto-volatility-forecast/notebooks/models/scaler.joblib'


# Map model names to their hardcoded absolute paths for easy lookup
MODEL_MAP = {
    'rf': RF_MODEL_PATH,
    'xgb': XGB_MODEL_PATH,
}


# Initialize Flask with hardcoded paths
app = Flask(__name__, 
            template_folder=TEMPLATE_DIR,
            static_folder=STATIC_DIR)

# Global variables to hold the loaded model objects
LOADED_MODELS = {}
GLOBAL_SCALER = None

# =========================================================
# 2. CENTRALIZED MODEL AND SCALER LOADING
# =========================================================

def load_models_and_scaler():
    """Load all trained models and the common scaler once at startup."""
    global GLOBAL_SCALER, LOADED_MODELS
    
    print("\n" + "="*70)
    print("PATH VERIFICATION & LOADING:")
    
    # 1. Load Scaler
    try:
        if os.path.exists(SCALER_PATH):
            GLOBAL_SCALER = joblib.load(SCALER_PATH)
            print(f"✅ Scaler loaded from: {SCALER_PATH}")
        else:
            print(f"🛑 CRITICAL ERROR: Scaler not found at {SCALER_PATH}. Exiting.")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading scaler: {e}")
        sys.exit(1)

    # 2. Load Models
    for alias, path in MODEL_MAP.items():
        try:
            if os.path.exists(path):
                model = joblib.load(path)
                LOADED_MODELS[alias] = model
                print(f"✅ Model '{alias}' loaded from: {path}")
            else:
                print(f"⚠️ Model '{alias}' not found at {path}. It won't be available.")
        except Exception as e:
            print(f"❌ Error loading model {alias}: {e}")
    
    # Check if any model was loaded
    if not LOADED_MODELS:
        print("🛑 No valid models were loaded. Using mock predictions only.")
    
    print("="*70)

# Call the loader function immediately when the application starts
load_models_and_scaler()

# =========================================================
# 3. HELPER FUNCTIONS (UNCHANGED LOGIC)
# =========================================================

def get_latest_data(ticker, days=60):
    """Fetch latest data for prediction"""
    try:
        df = yf.download(ticker, period=f"{days}d", interval="1d", progress=False) 
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None

def engineer_features(df):
    """Create same features as training"""
    try:
        df['Return'] = df['Close'].pct_change()
        df['Volatility_7d'] = df['Return'].rolling(7).std() 
        df['Volatility_14d'] = df['Return'].rolling(14).std()
        df['Price_Mean_7d'] = df['Close'].rolling(7).mean()
        df['Price_Mean_14d'] = df['Close'].rolling(14).mean()
        df['Price_Momentum'] = df['Close'] - df['Close'].shift(7)
        df['Return_Lag_1'] = df['Return'].shift(1)
        df['Return_Lag_3'] = df['Return'].shift(3)
        df['HL_Range'] = df['High'] - df['Low']
        df['Log_Volume'] = np.log1p(df['Volume'])
        
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"❌ Error engineering features: {e}")
        return df

def generate_mock_prediction(ticker):
    """Generate mock predictions when model is not available"""
    mock_data = {
        'BTC-USD': {'price': 45000, 'vol': 3.5},
        'ETH-USD': {'price': 2500, 'vol': 4.2},
        'BNB-USD': {'price': 320, 'vol': 5.1},
        'SOL-USD': {'price': 110, 'vol': 6.8}
    }
    data = mock_data.get(ticker, {'price': 1000, 'vol': 4.0})
    current_price = data['price']
    predicted_vol = data['vol'] / 100
    price_change = current_price * predicted_vol * 1.96
    
    result = {
        'model_used': 'MOCK',
        'current_price': current_price,
        'predicted_volatility': data['vol'],
        'current_volatility': data['vol'] - 0.5,
        'price_range_lower': round(current_price - price_change, 2),
        'price_range_upper': round(current_price + price_change, 2),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    return result

def predict_volatility(ticker, model_alias):
    """Make volatility prediction for a crypto using the specified model"""
    
    model = LOADED_MODELS.get(model_alias)
    
    if model is None or GLOBAL_SCALER is None:
        print(f"Model '{model_alias}' or scaler not available. Using mock predictions.")
        return generate_mock_prediction(ticker)
    
    df = get_latest_data(ticker)
    if df is None or len(df) < 30:
        print("Insufficient data, using mock predictions...")
        return generate_mock_prediction(ticker)
    
    df = engineer_features(df)
    
    if len(df) == 0:
        print("No data after feature engineering, using mock predictions...")
        return generate_mock_prediction(ticker)
    
    feature_cols = ['Close','Price_Mean_7d','Price_Mean_14d',
                    'Price_Momentum','Return_Lag_1','Return_Lag_3',
                    'HL_Range','Log_Volume']
    
    try:
        X = df[feature_cols].iloc[-1:].values
        X = X.reshape(1, -1)
        
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        X_scaled = GLOBAL_SCALER.transform(X)
        predicted_vol = float(model.predict(X_scaled)[0])  # Convert to Python float
        
        current_price = float(df['Close'].iloc[-1])
        current_vol = float(df['Volatility_7d'].iloc[-1])
        
        price_change = current_price * predicted_vol * 1.96
        lower_bound = current_price - price_change
        upper_bound = current_price + price_change
        
        result = {
            'model_used': model_alias.upper(),
            'current_price': float(round(current_price, 2)),
            'predicted_volatility': float(round(predicted_vol * 100, 2)),
            'current_volatility': float(round(current_vol * 100, 2)) if not pd.isna(current_vol) else 0.0,
            'price_range_lower': float(round(max(0, lower_bound), 2)), 
            'price_range_upper': float(round(upper_bound, 2)),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"✅ Prediction made by {model_alias} for {ticker}: Volatility = {result['predicted_volatility']}%")
        return result
        
    except Exception as e:
        print(f"❌ Error during prediction using {model_alias}: {e}")
        return generate_mock_prediction(ticker)

@app.route('/')
def home():
    """Render main page"""
    try:
        return render_template('index.html', available_models=list(MODEL_MAP.keys()))
    except Exception as e:
        return f"""
        <h1>Template Error</h1>
        <p>Could not find index.html</p>
        <p>Looking in: {TEMPLATE_DIR}</p>
        <p>Error: {str(e)}</p>
        """, 500

@app.route('/predict/<crypto>/<model_name>')
def predict(crypto, model_name):
    """API endpoint for predictions, accepts model_name parameter"""
    
    print(f"\n{'='*50}")
    print(f"Prediction requested for: {crypto} using model: {model_name}")
    print(f"{'='*50}")
    
    model_alias = model_name.lower()
    if model_alias not in MODEL_MAP:
        return jsonify({'error': f"Invalid model specified. Choose from: {list(MODEL_MAP.keys())}"}), 400

    ticker_map = {'BTC': 'BTC-USD', 'ETH': 'ETH-USD', 'BNB': 'BNB-USD', 'SOL': 'SOL-USD'}
    ticker = ticker_map.get(crypto.upper())
    
    if not ticker:
        return jsonify({'error': 'Invalid cryptocurrency ticker'}), 400
    
    try:
        prediction = predict_volatility(ticker, model_alias)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/historical/<crypto>')
def get_historical(crypto):
    """Get historical volatility data for charts"""
    ticker_map = {'BTC': 'BTC-USD', 'ETH': 'ETH-USD', 'BNB': 'BNB-USD', 'SOL': 'SOL-USD'}
    ticker = ticker_map.get(crypto.upper())
    
    if not ticker:
        return jsonify({'error': 'Invalid cryptocurrency'}), 400
    
    try:
        df = get_latest_data(ticker, days=90)
        if df is None or len(df) < 10:
            return jsonify({'error': 'Failed to fetch sufficient data'}), 500
        
        df = engineer_features(df)
        
        if len(df) == 0:
            return jsonify({'error': 'No data available'}), 500
        
        dates = df['Date'].dt.strftime('%Y-%m-%d').tolist()
        volatility = [float(v) for v in (df['Volatility_7d'] * 100).fillna(0).tolist()]
        prices = [float(p) for p in df['Close'].tolist()]
        
        result = {
            'dates': dates[-30:],
            'volatility': volatility[-30:],
            'prices': prices[-30:]
        }
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Route not found'}), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Starting Flask Application...")
    print("="*70)
    print(f"Template folder: {TEMPLATE_DIR}")
    print(f"Static folder: {STATIC_DIR}")
    print(f"Server will run at: http://localhost:5000")
    print("="*70 + "\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')
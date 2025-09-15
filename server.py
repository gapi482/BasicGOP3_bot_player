# server.py
from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

@app.route('/calculator', methods=['POST'])
def calculate():
    data = request.get_json()
    command = data.get('command', '')
    
    try:
        # Use pbots_calc for professional odds calculation
        result = subprocess.run(['pbots_calc', command], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            # Parse the result
            output = result.stdout.strip()
            # Extract equity value (simplified parsing)
            equity = float(output.split()[-1])
            return jsonify([(command, equity)])
        else:
            return jsonify([(command, 0.5)])  # Default equity
            
    except Exception as e:
        return jsonify([(command, 0.5)])  # Default equity

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
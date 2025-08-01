#!/usr/bin/env python3
"""
Production web server startup script
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Change to web directory
web_path = Path(__file__).parent / "web" 
os.chdir(web_path)

# Import and run the app
from app import socketio, app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🌐 Starting Web Audio Security Monitor on port {port}")
    print("=" * 50)
    
    socketio.run(app, debug=False, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)
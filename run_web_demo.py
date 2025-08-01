#!/usr/bin/env python3
"""
Quick demo script for web application
"""
import sys
import webbrowser
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    import os
    web_path = Path(__file__).parent / "web"
    sys.path.append(str(web_path))
    os.chdir(web_path)
    
    from app import socketio, app
    
    print("🌐 Starting Web Audio Monitor...")
    print("=" * 50)
    print("Server will start at http://localhost:5000")
    print("The web interface will open automatically")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Open browser after short delay
    def open_browser():
        time.sleep(2)
        try:
            webbrowser.open('http://localhost:5000')
            print("✅ Opened browser automatically")
        except:
            print("⚠️  Please manually open: http://localhost:5000")
    
    import threading
    threading.Thread(target=open_browser, daemon=True).start()
    
    socketio.run(app, debug=False, host='0.0.0.0', port=5000)
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install -r requirements.txt")
except KeyboardInterrupt:
    print("\n⏹️  Server stopped by user")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Check that port 5000 is available")
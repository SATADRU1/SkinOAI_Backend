#!/usr/bin/env python3
"""
Startup script for SkinOAI Backend Server
"""

import os
import sys
import logging
from app import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Start the SkinOAI backend server"""
    print("=" * 50)
    print("🚀 Starting SkinOAI Backend Server")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("❌ Error: app.py not found. Please run this script from the Backend directory.")
        sys.exit(1)
    
    print("📋 Server Configuration:")
    print(f"   Host: 0.0.0.0")
    print(f"   Port: 5000")
    print(f"   Debug: True")
    print(f"   API Endpoints:")
    print(f"     - GET  /       - Health check")
    print(f"     - GET  /ping   - Ping test")
    print(f"     - POST /predict - Skin prediction")
    print()
    
    print("🔄 Loading models...")
    try:
        # This will trigger model loading
        from model import load_model
        load_model()
        print("✅ Image classification model loaded")
    except Exception as e:
        print(f"⚠️  Warning: Image model loading failed: {e}")
    
    try:
        from TextModel import TextModel
        print("✅ TinyLlama text model initialized")
    except Exception as e:
        print(f"⚠️  Warning: TinyLlama model loading failed: {e}")
    
    print()
    print("🌐 Server starting...")
    print("   Access the API at:")
    print("     - Local: http://localhost:5000")
    print("     - Network: http://192.168.0.140:5000")
    print("   Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Quick start script for the Diabetes Prediction API
"""
import uvicorn
import sys
import os

def main():
    """Start the FastAPI server with optimal configuration"""
    print("🩺 Starting Diabetes Prediction API...")
    print("=" * 50)
    
    # Check if model file exists
    if not os.path.exists("diabetes_best_model.pkl"):
        print("❌ Error: Model file 'diabetes_best_model.pkl' not found!")
        print("Please ensure the model file is in the same directory as this script.")
        sys.exit(1)
    
    print("✅ Model file found")
    print("🚀 Starting server on http://127.0.0.1:8001")
    print("📚 API Documentation: http://127.0.0.1:8001/docs")
    print("🧪 Test Interface: Open test_client.html in your browser")
    print("=" * 50)
    print("Press Ctrl+C to stop the server")
    print()
    
    try:
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8001,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
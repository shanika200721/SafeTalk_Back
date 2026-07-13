#!/usr/bin/env python3
"""
Run the Suicide Prevention Agent API server
Usage: python run_server.py
"""

import os
import sys
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

def main():
    """Start the FastAPI server"""
    
    # Get configuration from environment
    environment = os.getenv("ENVIRONMENT", "development")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = environment == "development"
    
    print(f"[*] Starting Suicide Prevention API...")
    print(f"[*] Environment: {environment}")
    print(f"[*] Host: {host}")
    print(f"[*] Port: {port}")
    print(f"[*] Auto-reload: {reload}")
    print()
    print("[*] API Documentation available at:")
    print(f"    - Swagger UI: http://{host}:{port}/api/docs")
    print(f"    - ReDoc: http://{host}:{port}/api/redoc")
    print()
    
    # Start the server
    try:
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\n[!] Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n[!] Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

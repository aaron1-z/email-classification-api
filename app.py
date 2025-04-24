# app.py
import uvicorn
from api import app # Import the FastAPI app instance from api.py

# Add this print statement to see if the script runs at all
print("app.py script started...")

if __name__ == "__main__":
    # Configuration for running locally
    # Host '0.0.0.0' makes it accessible on your network
    print("Attempting to start Uvicorn server from app.py...") # Add this print
    uvicorn.run(
        "api:app",      # Tells uvicorn where the FastAPI app instance is (in api.py, named app)
        host="0.0.0.0", # Listen on all available network interfaces
        port=8000,      # Use port 8000
        reload=True     # Enable auto-reload for development (useful if you change code)
    )
    print("Uvicorn server should have started (or failed with an error above).") # Add this print
else:
     print("app.py was imported, not run directly.") # Add this print
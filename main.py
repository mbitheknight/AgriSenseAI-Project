from api import app  # Import FastAPI app instance
import uvicorn
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Get the port from environment variable
    uvicorn.run(app, host="0.0.0.0", port=port)

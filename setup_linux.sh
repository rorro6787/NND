#!/bin/bash

# Create the virtual environment
# Check if the 'venv' folder exists
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
  source venv/bin/activate
  pip install -r src/backend/requirements.txt
else
  source venv/bin/activate
  echo "Virtual environment already exists."
fi

# Launch the backend
echo "Launching backend..."
cd src/backend
# Assuming the backend is run with 'python app.py' or some equivalent command
python app.py &  # The '&' symbol runs the backend in the background
BACKEND_PID=$!   # Save the backend's PID to stop it later

# Go back to the main directory
cd ../

# Launch the frontend
echo "Launching frontend..."
cd frontend

npm ci
npm run dev &  # Run the frontend in the background
FRONTEND_PID=$!  # Save the frontend's PID to stop it later

# Function to stop the processes when the script finishes
cleanup() {
    echo "Stopping backend..."
    kill $BACKEND_PID
    echo "Stopping frontend..."
    kill $FRONTEND_PID
}

# Catch the interrupt signal (Ctrl+C) to run the cleanup
trap cleanup EXIT

# Keep the script running while the backend and frontend are active
wait $BACKEND_PID
wait $FRONTEND_PID

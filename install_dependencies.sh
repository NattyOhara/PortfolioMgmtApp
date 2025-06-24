#!/bin/bash

echo "Installing required dependencies for Portfolio Management App..."
echo

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Virtual environment not found. Creating one..."
    python -m venv venv
    source venv/bin/activate
fi

echo
echo "Installing Google Generative AI library..."
pip install google-generativeai

echo
echo "Installing Google API Python Client..."
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib

echo
echo "Installing BeautifulSoup4 and requests (if not already installed)..."
pip install beautifulsoup4 requests

echo
echo "Installing all requirements from requirements.txt..."
pip install -r requirements.txt

echo
echo "Installation complete!"
echo
echo "Please make sure to set the following environment variables in your .env file:"
echo "- GOOGLE_API_KEY"
echo "- GOOGLE_SEARCH_ENGINE_ID"
echo "- GEMINI_API_KEY"
echo
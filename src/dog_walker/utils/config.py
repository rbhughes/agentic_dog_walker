from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project settings
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# API endpoints (all free services)
OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
OPENROUTESERVICE_API_KEY = os.getenv("OPENROUTESERVICE_API_KEY", "")
OPENROUTESERVICE_URL = "https://api.openrouteservice.org/v2/matrix/foot-walking"

# Ollama configuration
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_BASE_URL = "http://localhost:11434"

# Agent settings
MAX_ITERATIONS = 10
TEMPERATURE = 0.1

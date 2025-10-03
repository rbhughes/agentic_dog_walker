# Agentic Dog Walker 🐕

An AI-powered dog walking route planner that uses LangChain and local LLMs to optimize routes with real street-based navigation.

## Features

- **Streamlit Web UI**: Modern web interface for managing pets and planning routes
- **Smart Route Optimization**: Uses real walking distances along actual streets (not straight-line)
- **Interactive Maps**: Generates HTML maps showing the optimized route with turn-by-turn paths
- **Weather Integration**: Checks weather conditions and provides safety recommendations
- **Local LLM**: Runs completely locally using Ollama (privacy-first)
- **Date Planning**: Select any future date for route planning

## Quick Start

1. **Install dependencies**:
```bash
uv sync
```

2. **Set up OpenRouteService API key**:
Create a `.env` file:
```
OPENROUTESERVICE_API_KEY=your_api_key_here
```
Get a free API key at https://openrouteservice.org/

3. **Install Ollama and pull the model**:
```bash
# Install Ollama from https://ollama.ai
ollama pull qwen2.5:14b
```

Note: We recommend `qwen2.5:14b` for best reliability. It's specifically optimized for tool use and works well on a MacBook Air with 24GB RAM.

4. **Install the package**:
```bash
uv pip install -e .
```

5. **Run the Streamlit app**:
```bash
uv run streamlit run app.py
```

## Using the Streamlit UI

1. **Add Pets**: Enter pet name, address, and walk duration
2. **Activate Pets**: Check the boxes for pets to include in today's route
3. **Select Date**: Choose the date for the walk (defaults to today)
4. **Plan Route**: Click "Plan Route" to optimize and generate the map
5. **View Results**: See the optimized route order, distance, time, and interactive map

## Architecture

### Agent Workflow

The agent follows a 4-step workflow:
1. **Geocode**: Convert addresses to coordinates using Nominatim
2. **Weather Check**: Assess conditions for the selected date
3. **Optimize Route**: Calculate optimal route using OR-Tools TSP solver with real walking distances
4. **Create Map**: Generate interactive HTML map with street-following paths

### Tools

- `geocode_addresses` - Converts addresses to lat/lon coordinates
- `check_weather` - Fetches weather forecast and provides recommendations
- `optimize_route` - Solves TSP with real street distances from OpenRouteService
- `create_route_map` - Generates Folium interactive map with route geometry

## Testing

The project includes comprehensive unit tests for all tools.

**Run all tests**:
```bash
uv run pytest
```

**Run specific test file**:
```bash
uv run pytest tests/test_geocoding.py
```

**Run with coverage**:
```bash
uv run pytest --cov=src/dog_walker
```

**Test files**:
- `tests/test_geocoding.py` - Address geocoding tests (8 tests)
- `tests/test_weather.py` - Weather API tests (8 tests)
- `tests/test_route_optimizer.py` - Route optimization tests (8 tests)
- `tests/test_mapping.py` - Map generation tests (8 tests)

All tests use mocking to avoid real API calls.

## Development

**Run the agent from command line**:
```bash
uv run python examples/test_agent.py
```

**Test individual tools**:
```bash
uv run python examples/test_basic.py
```

**Configuration**:
Edit `src/dog_walker/utils/config.py` to change:
- Ollama model (default: `qwen2.5:14b`)
- Max agent iterations (default: 15)
- Temperature (default: 0.0)

## Output Files

- **Maps**: `output/route_map_TIMESTAMP.html` - Interactive maps with route visualization
- **Route Plans**: `output/route_plan_TIMESTAMP.json` - Detailed route data (when using command line)
- **Pet Data**: `data/pets.csv` - Persistent storage of pet information (Streamlit UI)

## Tech Stack

- **LangChain** - Agent orchestration and ReAct pattern
- **Ollama** - Local LLM inference (Qwen 2.5 14B)
- **OR-Tools** - Route optimization (TSP solver)
- **OpenRouteService** - Real street distances and routing geometry
- **Nominatim** - Free address geocoding (OpenStreetMap)
- **Folium** - Interactive Leaflet.js maps
- **Streamlit** - Web UI framework
- **pytest** - Testing framework

## Model Recommendations

For MacBook Air with 24GB RAM:
- **Best**: `qwen2.5:14b` - Excellent for agentic tasks, tool use
- **Alternative**: `mistral:7b-instruct` - Faster, smaller, still capable
- **Experimental**: `llama3.2:11b-vision-instruct` - If you want multimodal features

The default `qwen2.5:14b` is specifically trained for function calling and has shown the best reliability in our testing.

## License

MIT

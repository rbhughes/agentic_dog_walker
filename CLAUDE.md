# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agentic Dog Walker is a LangChain-based application that uses local LLMs (via Ollama) to optimize dog walking routes. It integrates multiple tools for geocoding, weather checking, and route optimization to plan efficient dog walking schedules with real street-based routing and interactive map visualization.

## Development Commands

**Start Ollama (required):**
```bash
ollama serve
```

**Run the agent:**
```bash
python examples/test_agent.py
```

**Run tool tests:**
```bash
python examples/test_basic.py
```

**Type checking:**
```bash
mypy src/
```

**Import sorting:**
```bash
isort src/
```

**Run tests:**
```bash
pytest
```

## Architecture

### Core Components

**LangChain Agent** (`src/dog_walker/agent.py`): 
- Class-based agent with state management (`DogWalkerAgent`)
- Orchestrates geocoding, weather checking, route optimization, and map generation
- Returns structured `RouteResult` objects
- Exports results to JSON
- Uses ReAct agent pattern with Ollama LLM backend

**Tools** (`src/dog_walker/tools/`):
- `geocoding.py`: Converts addresses to coordinates using Nominatim (OpenStreetMap) API
- `weather.py`: Fetches weather data from Open-Meteo and generates dog-walking safety recommendations
- `route_optimizer.py`: Uses OR-Tools TSP solver with real walking distances from OpenRouteService
- `mapping.py`: Creates interactive Folium HTML maps with street-following routes

**Configuration** (`src/dog_walker/utils/config.py`): Centralized settings including:
- Ollama model: `llama3.1:8b` (localhost:11434)
- OpenRouteService API key loaded from `.env`
- Free APIs: Nominatim for geocoding, Open-Meteo for weather
- Project paths for data and output directories

### Key Technical Details

**Route Optimization**: Uses OR-Tools' TSP solver with:
- Real walking distances from OpenRouteService Matrix API (not straight-line)
- TSP solving with optional time windows for appointment constraints
- Returns `locations_for_map` array pre-ordered for visualization
- Walking speed assumption: 5 km/h + visit duration
- Includes return trip to starting location

**Map Visualization**: Creates interactive HTML maps with:
- Street-following routes from OpenRouteService Directions API
- Numbered stop markers with pet names and durations
- Color-coded routes (blue for segments, dashed green for return home)
- Popup details on markers showing pet info and visit duration

**Weather Assessment**: Checks hourly data for temperature, precipitation, wind speed and generates safety recommendations for dogs

**Agent Flow**: The agent autonomously:
1. Geocodes addresses using Nominatim
2. Optimizes routes with real street distances
3. Creates interactive maps showing the optimized route
4. Optionally checks weather conditions
5. Returns structured results with all planning details

## Dependencies

- **LangChain**: Agent framework with ReAct pattern
- **Ollama**: Local LLM backend (must be running on localhost:11434)
- **OR-Tools**: TSP route optimization solver
- **OpenRouteService**: Real street-based walking distances and routes (API key required in `.env`)
- **Nominatim**: Free address geocoding (OpenStreetMap)
- **Open-Meteo**: Free weather API
- **Folium**: Interactive map visualization
- **uv**: Package manager

## Environment Setup

Create a `.env` file with:
```
OPENROUTESERVICE_API_KEY=your_api_key_here
```

The OpenRouteService API key is required for route optimization and map generation.

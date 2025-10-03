# Agentic Dog Walker 🐕

An AI-powered dog walking route planner that uses LangChain and local LLMs to optimize routes with real street-based navigation.

## Features

- **Natural Language Planning**: Describe your dog walking schedule in plain English
- **Smart Route Optimization**: Uses real walking distances along actual streets (not straight-line)
- **Interactive Maps**: Generates HTML maps showing the optimized route with turn-by-turn paths
- **Weather Integration**: Checks weather conditions and provides safety recommendations
- **Local LLM**: Runs completely locally using Ollama (privacy-first)

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

3. **Start Ollama**:
```bash
ollama serve
```

Make sure you have the model: `ollama pull llama3.1:8b`

4. **Run the agent**:
```bash
uv run python examples/test_agent.py
```

## Example Usage

The agent understands natural language requests like:

> "I need to plan a dog walking route for today in Chicago. Here are my visits:
> - Max at 108 N State St, 30 minutes
> - Bella at 2001 N Clark St, 20 minutes
> - Charlie at 1060 W Addison St, 25 minutes
> - Luna at 201 E Randolph St, 15 minutes
>
> Please optimize the route and create a map."

The agent will:
1. Geocode all addresses
2. Optimize the route using real walking distances
3. Generate an interactive HTML map
4. Provide time estimates and recommendations

## Output

- **Console**: Route summary with stop order and timing
- **JSON**: Detailed results exported to `output/route_plan_*.json`
- **HTML Map**: Interactive map saved to `output/route_map_*.html`

## Tech Stack

- **LangChain** - Agent orchestration
- **Ollama** - Local LLM (llama3.1:8b)
- **OR-Tools** - Route optimization (TSP solver)
- **OpenRouteService** - Real street distances and routing
- **Nominatim** - Address geocoding
- **Folium** - Interactive maps

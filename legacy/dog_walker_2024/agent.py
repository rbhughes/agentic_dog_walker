from typing import Any
from dataclasses import dataclass, field
from langchain_ollama import OllamaLLM
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool

from dog_walker.tools.geocoding import geocoding_tool
from dog_walker.tools.weather import weather_tool
from dog_walker.tools.route_optimizer import route_optimizer_tool
from dog_walker.tools.mapping import mapping_tool
from dog_walker.utils.config import (
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    MAX_ITERATIONS,
    TEMPERATURE,
    OUTPUT_DIR,
)


@dataclass
class RouteResult:
    """Structured result from a dog walking route planning session."""

    query: str
    addresses: list[str] = field(default_factory=list)
    geocoded_locations: list[dict[str, Any]] = field(default_factory=list)
    weather_assessment: str = ""
    optimized_route: dict[str, Any] = field(default_factory=dict)
    final_answer: str = ""
    agent_output: str = ""


class DogWalkerAgent:
    """
    Dog walking route planning agent with state management and structured results.

    This agent orchestrates geocoding, weather checking, and route optimization
    to help dog walkers plan efficient routes.
    """

    AGENT_PROMPT: str = """You are a dog walking route planner. Follow the steps exactly.

{tools}

Tool names: {tool_names}

STEPS (complete ALL 4 steps before giving Final Answer):
1. geocode_addresses - convert addresses to coordinates
2. check_weather - check weather at first location
3. optimize_route - MUST wrap visits in object: {{"visits": [...]}}
4. create_route_map - create map using EXACT JSON from step 3

RULES:
- Do NOT give Final Answer until ALL 4 steps are complete
- For create_route_map: Action Input = exact JSON from optimize_route (no extra text)
- Final Answer = human-readable summary (NOT JSON)
- Weather only noteworthy if: temp < 4°C or > 32°C, rain/snow, wind > 32 km/h

FORMAT:
Question: [question]
Thought: [what to do next]
Action: [tool name]
Action Input: [input]
Observation: [result]
... repeat until all 4 steps done ...
Thought: All 4 steps complete
Final Answer: [human summary: route order, distance, time, noteworthy weather if any]

EXAMPLES:
Step 1:
Action: geocode_addresses
Action Input: ["123 Main St, Chicago, IL", "456 Oak Ave, Boston, MA"]

Step 3:
Action: optimize_route
Action Input: {{"visits": [{{"pet_name": "Max", "coordinates": [41.88, -87.63], "duration": 30}}]}}

NOTE: Use JSON format (double quotes), NOT Python format (single quotes)

Question: {input}
Thought: {agent_scratchpad}"""

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = TEMPERATURE,
        max_iterations: int = MAX_ITERATIONS,
    ):
        """
        Initialize the dog walking agent.

        Args:
            model: Ollama model name
            base_url: Ollama server URL
            temperature: LLM temperature for generation
            max_iterations: Maximum agent iterations
        """
        self.model: str = model
        self.base_url: str = base_url
        self.temperature: float = temperature
        self.max_iterations: int = max_iterations

        # Initialize LLM
        self.llm: OllamaLLM = OllamaLLM(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
        )

        # Initialize tools
        self.tools: list[BaseTool] = [
            geocoding_tool,
            weather_tool,
            route_optimizer_tool,
            mapping_tool,
        ]

        # Create agent
        prompt = PromptTemplate.from_template(self.AGENT_PROMPT)
        agent = create_react_agent(self.llm, self.tools, prompt)

        # Create executor with better error handling
        self.agent_executor: AgentExecutor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=self.max_iterations,
            handle_parsing_errors="Check your output and make sure it conforms to the format instructions!",
            return_intermediate_steps=True,
        )

        # Session state
        self.current_result: RouteResult | None = None

    def plan_route(self, query: str) -> RouteResult:
        """
        Plan a dog walking route based on the query.

        Args:
            query: User's route planning request

        Returns:
            Structured RouteResult with all planning details
        """
        # Initialize result
        result = RouteResult(query=query)

        # Execute agent
        agent_response = self.agent_executor.invoke({"input": query})
        result.agent_output = agent_response["output"]
        result.final_answer = agent_response["output"]

        # Store current result
        self.current_result = result

        return result

    def get_current_result(self) -> RouteResult | None:
        """Get the most recent route planning result."""
        return self.current_result

    def export_result_to_file(self, filename: str | None = None) -> str:
        """
        Export the current result to a JSON file.

        Args:
            filename: Optional filename, defaults to timestamped file

        Returns:
            Path to the exported file
        """
        import json
        from datetime import datetime

        if self.current_result is None:
            raise ValueError("No result to export")

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"route_plan_{timestamp}.json"

        filepath = OUTPUT_DIR / filename

        with open(filepath, "w") as f:
            json.dump(
                {
                    "query": self.current_result.query,
                    "addresses": self.current_result.addresses,
                    "geocoded_locations": self.current_result.geocoded_locations,
                    "weather_assessment": self.current_result.weather_assessment,
                    "optimized_route": self.current_result.optimized_route,
                    "final_answer": self.current_result.final_answer,
                },
                f,
                indent=2,
            )

        return str(filepath)

    def reset(self) -> None:
        """Reset the agent state."""
        self.current_result = None


def create_dog_walker_agent(**kwargs: Any) -> DogWalkerAgent:
    """
    Factory function to create a DogWalkerAgent instance.

    Args:
        **kwargs: Optional arguments to pass to DogWalkerAgent constructor

    Returns:
        Configured DogWalkerAgent instance
    """
    return DogWalkerAgent(**kwargs)

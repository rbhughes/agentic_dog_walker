from typing import Any
from dataclasses import dataclass, field
from langchain_ollama import OllamaLLM
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

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

    AGENT_PROMPT = """You are a helpful dog walking assistant that helps optimize routes for dog walkers.

You have access to the following tools:

{tools}

Tool names: {tool_names}

WORKFLOW: When planning a route, follow these steps:
1. Use geocode_addresses to convert addresses to coordinates
2. Use optimize_route with the geocoded data - it returns JSON with "locations_for_map"
3. CRITICAL: To create map, take the COMPLETE JSON from optimize_route and pass it directly to create_route_map
   DO NOT create your own locations or route_sequence arrays
4. Optionally use check_weather

IMPORTANT:
- For create_route_map: Pass the complete optimize_route output JSON as-is
- DO NOT manually build locations or route_sequence for the map

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

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
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_iterations = max_iterations

        # Initialize LLM
        self.llm = OllamaLLM(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
        )

        # Initialize tools
        self.tools = [
            geocoding_tool,
            weather_tool,
            route_optimizer_tool,
            mapping_tool,
        ]

        # Create agent
        prompt = PromptTemplate.from_template(self.AGENT_PROMPT)
        agent = create_react_agent(self.llm, self.tools, prompt)

        # Create executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=self.max_iterations,
            handle_parsing_errors=True,
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

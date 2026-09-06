#!/usr/bin/env python3

import sys

sys.path.append("src")

from dog_walker.agent import DogWalkerAgent


def test_agent():
    """Test the dog walking agent with a sample query."""

    # Create the agent
    print("Creating dog walking agent...")
    agent = DogWalkerAgent()

    # Test query
    query = """
    I need to plan a dog walking route for tomorrow (2025-10-03) in Chicago.
    Here are my visits:
    - Max at 108 N State St, 30 minutes
    - Bella at 2001 N Clark St, 20 minutes
    - Charlie at 1060 W Addison St, 25 minutes
    - Luna at 201 E Randolph St, 15 minutes

    Please geocode the addresses, optimize the route, and check the weather.
    """

    print(f"\nQuery: {query}\n")
    print("Running agent (this may take a moment)...\n")

    # Run the agent
    result = agent.plan_route(query)

    print("\n" + "=" * 80)
    print("AGENT RESULT")
    print("=" * 80)
    print(f"\nFinal Answer:\n{result.final_answer}\n")

    # Export result
    filepath = agent.export_result_to_file()
    print(f"Result exported to: {filepath}")


if __name__ == "__main__":
    test_agent()

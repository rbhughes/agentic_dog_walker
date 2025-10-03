#!/usr/bin/env python3
"""Streamlit UI for managing dog walking visits."""
# pyright: reportUnusedCallResult=false

import json
import os
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from dog_walker.agent import DogWalkerAgent
from dog_walker.utils.config import OUTPUT_DIR

# Configure page to be wide
st.set_page_config(page_title="Dog Walking Planner", page_icon="🐕", layout="wide")

# File to store pet data
DATA_FILE = Path("data/pets.csv")
DATA_FILE.parent.mkdir(exist_ok=True)


def load_pets() -> pd.DataFrame:
    """Load pets from CSV file."""
    if DATA_FILE.exists():
        return pd.read_csv(DATA_FILE)
    else:
        # Create empty dataframe with correct columns
        data: dict[str, list[str | bool | int]] = {
            "active": [],
            "pet_name": [],
            "address": [],
            "duration": [],
        }
        return pd.DataFrame(data)


def save_pets(df: pd.DataFrame) -> None:
    """Save pets to CSV file."""
    df.to_csv(DATA_FILE, index=False)


def add_pet(pet_name: str, address: str, duration: int) -> None:
    """Add a new pet to the CSV file."""
    df = load_pets()
    new_pet = pd.DataFrame(
        [
            {
                "active": True,
                "pet_name": pet_name,
                "address": address,
                "duration": duration,
            }
        ]
    )
    df = pd.concat([df, new_pet], ignore_index=True)
    save_pets(df)


def remove_pet(index: int) -> None:
    """Remove a pet from the CSV file."""
    df = load_pets()
    df = df.drop(index).reset_index(drop=True)
    save_pets(df)


def toggle_active(index: int) -> None:
    """Toggle the active status of a pet."""
    df = load_pets()
    df.at[index, "active"] = not df.at[index, "active"]
    save_pets(df)


def main():
    st.title("🐕 Dog Walking Planner")

    # Add new pet form
    st.subheader("Add New Pet")

    with st.form("add_pet_form", clear_on_submit=True):
        col1, col2, col3 = st.columns([2, 3, 1])

        with col1:
            pet_name = st.text_input("Pet Name", placeholder="Max")

        with col2:
            address = st.text_input("Address", placeholder="123 Main St, City, State")

        with col3:
            duration = st.text_input("Duration (min)", placeholder="30")

        submitted = st.form_submit_button("Add Pet", use_container_width=True)

        if submitted:
            # Validation
            errors = []

            if not pet_name.strip():
                errors.append("Pet name is required")

            if not address.strip():
                errors.append("Address is required")

            if not duration.strip():
                errors.append("Duration is required")
            else:
                try:
                    duration_int = int(duration)
                    if duration_int <= 0:
                        errors.append("Duration must be a positive number")
                except ValueError:
                    errors.append("Duration must be a valid number")

            if errors:
                for error in errors:
                    st.error(error)
            else:
                add_pet(pet_name.strip(), address.strip(), int(duration))
                st.success(f"Added {pet_name}!")
                st.rerun()

    # Display pets
    st.subheader("Scheduled Visits")

    df = load_pets()

    if df.empty:
        st.info("No pets scheduled yet. Add one above!")
    else:
        # Display each pet as a row
        for idx, row in df.iterrows():
            col1, col2, col3, col4, col5 = st.columns([0.5, 2, 4, 1.5, 1])

            with col1:
                # Checkbox for active status
                active = st.checkbox(
                    "Active",
                    value=bool(row["active"]),
                    key=f"active_{idx}",
                    label_visibility="collapsed",
                )
                if active != bool(row["active"]):
                    toggle_active(idx)  # pyright: ignore[reportArgumentType]
                    st.rerun()

            with col2:
                st.write(row["pet_name"])

            with col3:
                st.write(row["address"])

            with col4:
                st.write(f"{int(row['duration'])} min")

            with col5:
                if st.button("Remove", key=f"remove_{idx}", use_container_width=True):
                    remove_pet(idx)  # pyright: ignore[reportArgumentType]
                    st.rerun()

        # Summary and Plan Route button
        st.divider()

        col1, col2, col3 = st.columns([2, 1.5, 1])

        with col1:
            active_count = df[df["active"]].shape[0]
            total_duration = df[df["active"]]["duration"].sum()
            st.write(
                f"**Active visits:** {active_count} | **Total duration:** {int(total_duration)} minutes"
            )

        with col2:
            selected_date = st.date_input(
                "Walk date",
                value=date.today(),
                key="walk_date",
                label_visibility="collapsed",
            )

        with col3:
            plan_route_btn = st.button(
                "🗺️ Plan Route", use_container_width=True, type="primary"
            )

        # Plan route when button is clicked
        if plan_route_btn:
            active_pets = df[df["active"]]

            if active_pets.empty:
                st.warning(
                    "No active visits to plan. Please activate at least one pet."
                )
            else:
                with st.spinner("Planning optimal route..."):
                    # Build query for agent
                    visit_list = []
                    for _, pet in active_pets.iterrows():
                        visit_list.append(
                            f"- {pet['pet_name']} at {pet['address']}, {int(pet['duration'])} minutes"
                        )

                    query = f"""
                    I need to plan a dog walking route for {selected_date.strftime("%Y-%m-%d")}. Here are my visits:
                    {chr(10).join(visit_list)}

                    Please geocode the addresses, optimize the route, check the weather for the date, and create a map.
                    If the weather is noteworthy (cold, rainy, or windy), include it in your final answer.
                    """

                    # Run agent
                    try:
                        agent = DogWalkerAgent()
                        result = agent.plan_route(query)

                        # Store result in session state
                        st.session_state["route_result"] = result
                        st.session_state["show_map"] = True

                        st.success("Route planned successfully!")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error planning route: {str(e)}")

    # Display map if available
    if st.session_state.get("show_map") and st.session_state.get("route_result"):
        st.subheader("📍 Optimized Route Map")

        result = st.session_state["route_result"]

        # Display route summary - format if it looks like JSON
        final_answer = result.final_answer

        # Check if the answer looks like JSON and try to parse it for better formatting
        if final_answer.strip().startswith("{"):
            try:
                data = json.loads(final_answer)

                # Format the route information nicely
                summary_parts = []

                if "visit_order" in data:
                    summary_parts.append("**Route Order:**")
                    for i, visit in enumerate(data["visit_order"], 1):
                        pet_name = visit.get("pet_name", "Unknown")
                        duration = visit.get("duration_minutes", "?")
                        summary_parts.append(f"{i}. {pet_name} ({duration} min)")

                if "total_distance_meters" in data:
                    distance_km = data["total_distance_meters"] / 1000
                    summary_parts.append(f"\n**Total Distance:** {distance_km:.2f} km")

                if "estimated_time_hours" in data:
                    summary_parts.append(
                        f"**Estimated Time:** {data['estimated_time_hours']:.1f} hours"
                    )

                st.info("\n".join(summary_parts))
            except (json.JSONDecodeError, KeyError):
                # If parsing fails, just display as-is
                st.info(final_answer)
        else:
            # Display as-is if not JSON
            st.info(final_answer)

        # Get the latest map file from output directory
        if OUTPUT_DIR.exists():
            map_files = list(OUTPUT_DIR.glob("route_map_*.html"))
            if map_files:
                # Get the most recent map file
                latest_map = max(map_files, key=os.path.getmtime)

                # Read and display the HTML map
                with open(latest_map, "r") as f:
                    map_html = f.read()

                # Display using HTML component
                components.html(map_html, height=600, scrolling=True)
            else:
                st.warning(
                    "No map file found. The agent may not have completed successfully."
                )
        else:
            st.warning(f"Output directory not found: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

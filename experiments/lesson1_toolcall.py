import json
import urllib.request
from datetime import UTC, datetime

# from zoneinfo import ZoneInfo  # stdlib, Python 3.9+

OLLAMA = "http://fossil:11434/api/chat"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_weather",
            "description": "Get the weather forecast for a city on a date.",
            "parameters": {
                # YOU write this: JSON Schema with properties city (string,
                # required) and date (string, ISO date, required)
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "urban area"},
                    "date": {"type": "date", "description": "ISO date YYYY-MM-DD"},
                },
                "required": ["city", "date"],
            },
        },
    }
]

now_utc = datetime.now(UTC)


def chat(messages):
    body = json.dumps(
        {
            "model": "qwen2.5:7b",
            "messages": messages,
            "tools": TOOLS,
            "stream": False,
            "options": {"num_thread": 10, "temperature": 0},
        }
    ).encode()
    req = urllib.request.Request(OLLAMA, body, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))["message"]


####################
blah = "Should I walk Rex in Calgary tomorrow around 4pm?"
# blah = "Should I walk Rex in Denver?"
#
# today = datetime.datetime.now()
today = datetime.now().astimezone()


msgs = [
    {
        "role": "system",
        "content": f"Today is {today:%A} {today.isoformat()}. Pass dates as ISO YYYY-MM-DD",
    },
    {"role": "user", "content": blah},
]
####################

reply = chat(msgs)
print("FIRST REPLY:", json.dumps(reply, indent=2))

# Round 2 — feed a fake result back and watch it become prose:
msgs.append(reply)
msgs.append(
    {
        "role": "tool",
        "content": json.dumps({"temp_c": -21, "precip_mm": 40, "wind_kph": 30}),
    }
)
print("FINAL:", json.dumps(chat(msgs), indent=2))

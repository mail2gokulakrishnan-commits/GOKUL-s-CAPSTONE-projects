import os
import json
import re
import requests
from dotenv import load_dotenv
from jsonschema import validate, ValidationError

# Load API Key

load_dotenv()
API_KEY = os.getenv("LLM_API_KEY")

if not API_KEY:
    raise ValueError("LLM_API_KEY not found in .env")

URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-4.1-mini"

# Sample Dataset

records = [
    {
        "ticket_id": "T001",
        "customer": "John Smith",
        "issue": "Battery drains within one hour after installing the latest Windows update.",
        "priority_note": "Customer is angry and requested urgent help."
    },
    {
        "ticket_id": "T002",
        "customer": "Sarah Lee",
        "issue": "Laptop screen flickers occasionally while using Zoom meetings.",
        "priority_note": "Normal priority."
    },
    {
        "ticket_id": "T003",
        "customer": "Michael Brown",
        "issue": "Several keyboard keys stopped working yesterday.",
        "priority_note": "Customer wants replacement if needed."
    }
]


# Expected JSON Schema

schema = {
    "type": "object",
    "properties": {
        "ticket_id": {"type": "string"},
        "issue_category": {"type": "string"},
        "priority": {"type": "string"},
        "sentiment": {"type": "string"},
        "needs_follow_up": {"type": "boolean"},
        "summary": {"type": "string"}
    },
    "required": [
        "ticket_id",
        "issue_category",
        "priority",
        "sentiment",
        "needs_follow_up",
        "summary"
    ]
}

# PII Guardrail


def has_pii(text):
    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    phone_pattern = r"\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"

    return bool(
        re.search(email_pattern, text)
        or re.search(phone_pattern, text)
    )


# LLM Function


def call_llm(system_prompt,
             user_prompt,
             temperature=0.0,
             max_tokens=512):

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    response = requests.post(
        URL,
        headers=headers,
        json=payload
    )

    if response.status_code != 200:
        print("Status Code:", response.status_code)
        print(response.text)
        return None

    return response.json()["choices"][0]["message"]["content"]


# Prompts


system_prompt = """
You are a structured data extraction assistant.

Return ONLY valid JSON.

Do not include markdown.

Required fields:

ticket_id
issue_category
priority
sentiment
needs_follow_up
summary
"""

def create_prompt(record):

    return f"""
Example 1

Input

{{
    "ticket_id":"T100",
    "customer":"Alice",
    "issue":"Laptop battery drains quickly",
    "priority_note":"Urgent"
}}

Output

{{
    "ticket_id":"T100",
    "issue_category":"Battery",
    "priority":"High",
    "sentiment":"Negative",
    "needs_follow_up":true,
    "summary":"Battery drains quickly."
}}

Example 2

Input

{{
    "ticket_id":"T101",
    "customer":"Bob",
    "issue":"WiFi disconnects occasionally",
    "priority_note":"Normal"
}}

Output

{{
    "ticket_id":"T101",
    "issue_category":"Network",
    "priority":"Medium",
    "sentiment":"Neutral",
    "needs_follow_up":false,
    "summary":"WiFi disconnects occasionally."
}}

Actual Input

{json.dumps(record, indent=4)}

Return ONLY JSON.
"""


# JSON Validation


def validate_json(response):

    fallback = {
        "ticket_id": None,
        "issue_category": None,
        "priority": None,
        "sentiment": None,
        "needs_follow_up": None,
        "summary": None
    }

    try:

        response = response.strip()

        data = json.loads(response)

        validate(instance=data, schema=schema)

        return data

    except json.JSONDecodeError as e:

        print("JSON Decode Error:", e)

    except ValidationError as e:

        print("Validation Error:", e)

    return fallback

# API Test

print("=" * 80)
print("Testing API")
print("=" * 80)

test = call_llm(
    "Reply with only the word hello.",
    "Say hello."
)

print(test)


# Guardrail Test


print("\n" + "=" * 80)
print("Guardrail Test")
print("=" * 80)

blocked = "Contact me at abc@gmail.com"

clean = "Laptop battery drains quickly."

print("Blocked Input:", has_pii(blocked))
print("Clean Input:", has_pii(clean))


# Main Pipeline


print("\n" + "=" * 80)
print("Structured Extraction")
print("=" * 80)

for record in records:

    user_input = json.dumps(record)

    if has_pii(user_input):
        print("Input blocked: PII detected.")
        continue

    prompt = create_prompt(record)

    raw = call_llm(
        system_prompt,
        prompt,
        temperature=0
    )

    validated = validate_json(raw)

    print("\n" + "-" * 80)

    print("INPUT")
    print(record)

    print("\nRAW RESPONSE")
    print(raw)

    print("\nVALIDATED OUTPUT")
    print(validated)

# Temperature Comparison


print("\n" + "=" * 80)
print("Temperature Comparison")
print("=" * 80)

for record in records:

    prompt = create_prompt(record)

    out0 = call_llm(
        system_prompt,
        prompt,
        temperature=0
    )

    out07 = call_llm(
        system_prompt,
        prompt,
        temperature=0.7
    )

    print("\nTicket:", record["ticket_id"])
    print("\nTemperature = 0")
    print(out0)

    print("\nTemperature = 0.7")
    print(out07)

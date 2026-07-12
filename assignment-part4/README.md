
# Part 4 – LLM Powered Feature (Track A)

## Chosen Track
**Track A – Structured JSON Extraction**

This project uses an LLM to convert customer support tickets into standardized JSON data.

---

## Technologies

- Python
- OpenRouter API
- requests
- python-dotenv
- jsonschema

---

## Prompt Design

### System Prompt

```text
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
```

### User Prompt Template

```text
Example 1

Input:
{Example Ticket}

Output:
{Expected JSON}

Example 2

Input:
{Example Ticket}

Output:
{Expected JSON}

Actual Input:
{ticket_record}

Return ONLY JSON.
```

Two few-shot examples are included before the actual record to encourage consistent structured output.

---

## Why Temperature = 0?

Temperature was set to **0** because this task requires deterministic JSON output. Lower temperatures always choose the highest-probability next token, producing consistent responses that are easier to validate. A higher temperature (0.7) introduces randomness, which may change wording or formatting.

---

## JSON Schema

Required fields:

- ticket_id (string)
- issue_category (string)
- priority (string)
- sentiment (string)
- needs_follow_up (boolean)
- summary (string)

All responses are parsed with `json.loads()` and validated using `jsonschema.validate()`.

If validation fails, the program returns a fallback dictionary with all fields set to `null`.

---

## PII Guardrail

Before every API call the input is checked for:

- Email addresses
- Phone numbers

If detected, the request is blocked and the API is not called.

Example:

| Input | Result |
|-------|--------|
| Contact me at abc@gmail.com | Blocked |
| Laptop battery drains quickly | Allowed |

---

## Temperature Comparison

| Input | Temp=0 | Temp=0.7 | Key Difference |
|------|---------|-----------|----------------|
| Ticket 1 | Consistent JSON | Similar JSON with wording variations | Higher temperature may vary summaries/categories |
| Ticket 2 | Consistent JSON | Slight wording differences | Less deterministic |
| Ticket 3 | Consistent JSON | Different phrasing | More variation |

---

## Demonstration Results

Replace the table below with your actual program output after running the script.

| Input | LLM Output | Valid JSON | Guardrail |
|------|------------|------------|-----------|
| Ticket 1 | ✔ | Pass | Pass |
| Ticket 2 | ✔ | Pass | Pass |
| Ticket 3 | ✔ | Pass | Pass |

---


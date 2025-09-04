# Simple Calculator (Streamlit)

## Overview

I built a tiny **Streamlit** app to practice shipping a clean, single-file UI with a pure-Python backend function. The app supports **Add, Subtract, Multiply, Divide** with basic validation for division by zero. It’s deliberately minimal and easy to read for quick code reviews.

## What’s inside

* **`calculator_app.py`** — Streamlit UI + a pure function `calculator(operation, num1, num2)` that does the math and returns the result. The UI has two number inputs, an operation dropdown, and a **Calculate** button that displays the answer.&#x20;

## Quickstart

```bash
# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -U pip
pip install streamlit

# run
streamlit run calculator_app.py
```

## How to use

1. Enter **two numbers**.
2. Choose an **operation** (Add / Subtract / Multiply / Divide).
3. Click **Calculate** → the result appears in a success toast.
4. Division by zero is handled gracefully with an error message.&#x20;

## Project structure

```
calculator-app/
├─ calculator_app.py
└─ README.md  ← (this file)
```

## Design notes (what I focused on)

* **Pure function first:** `calculator(...)` is independent of the UI, so it’s easy to test and reuse.&#x20;
* **One-screen UI:** simple title, two numeric inputs defaulting to `0.0`, a selectbox for the operation, and a single action button.&#x20;
* **User feedback:** results are shown with `st.success(...)` so the answer is visually distinct.&#x20;

## (Optional) Tests

If I add tests, I’ll start with a tiny `pytest` file:

```python
# tests/test_calculator.py
from calculator_app import calculator

def test_add():        assert calculator("Add", 2, 3) == 5
def test_subtract():   assert calculator("Subtract", 5, 2) == 3
def test_multiply():   assert calculator("Multiply", 4, 3) == 12
def test_divide():     assert calculator("Divide", 8, 2) == 4
def test_divide_zero():assert "error" in str(calculator("Divide", 8, 0)).lower()
```

Run with:

```bash
pip install pytest
pytest -q
```

## Roadmap

* Add keyboard shortcuts & inline result updating (no button needed).
* Format numbers (fixed decimals) and improve error text.
* Expand operations: power, modulus, roots; optional history panel.
* Add unit tests + pre-commit hooks (black, ruff) for quick quality checks.
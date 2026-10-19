"""Constants for Auto-Heal Sandbox Agent UI, presets, and styles."""

# ==============================================================================
# 1. Preset Bug Samples
# ==============================================================================

# Preset 1: Basic Calculator
BASIC_BUGGY_CODE = '''class Calculator:
    def add(self, a: float, b: float) -> float:
        return a + b

    def divide(self, a: float, b: float) -> float:
        # BUG: Doesn't raise ValueError on division by zero
        return a / b

    def average(self, numbers: list) -> float:
        # BUG: Off-by-one division in denominator
        if not numbers:
            return 0.0
        return sum(numbers) / (len(numbers) - 1)
'''

BASIC_TEST_CODE = '''import pytest

try:
    from solution import Calculator
except ImportError:
    from buggy_calculator import Calculator

def test_divide():
    calc = Calculator()
    assert calc.divide(10, 2) == 5.0
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        calc.divide(10, 0)

def test_average():
    calc = Calculator()
    assert calc.average([10, 20, 30]) == 20.0
    assert calc.average([]) == 0.0
'''

BASIC_TASK_DESC = (
    "Fix Calculator bugs: 1. Raise ValueError with 'Cannot divide by zero' on division by zero. "
    "2. Fix off-by-one division in average calculation."
)

# Preset 2: Advanced LRU Cache with TTL (in English)
ADVANCED_BUGGY_CODE = '''import time

class LRUCacheWithTTL:
    def __init__(self, capacity: int, default_ttl: float = 60.0):
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache = {}          # key -> value
        self.expiry = {}         # key -> timestamp
        self.access_order = []   # list of keys (oldest to newest)

    def _is_expired(self, key: str) -> bool:
        if key in self.expiry:
            return time.time() > self.expiry[key]
        return False

    def get(self, key: str):
        # BUG 1: Does not delete expired keys from cache, expiry, or access_order
        if key not in self.cache or self._is_expired(key):
            return None

        # BUG 2: Misses updating the key position in access_order on cache hit
        return self.cache[key]

    def put(self, key: str, value, ttl: float = None) -> None:
        current_ttl = ttl if ttl is not None else self.default_ttl
        expiry_time = time.time() + current_ttl

        # BUG 3: If key already exists, does not remove old position in access_order leading to duplicates
        if key not in self.cache:
            if len(self.cache) >= self.capacity:
                # BUG 4: Evicts newest item (-1) instead of oldest item (0)
                lru_key = self.access_order.pop(-1)
                del self.cache[lru_key]
                if lru_key in self.expiry:
                    del self.expiry[lru_key]

        self.cache[key] = value
        self.expiry[key] = expiry_time
        self.access_order.append(key)
'''

ADVANCED_TEST_CODE = '''import time
import pytest
from solution import LRUCacheWithTTL

def test_lru_eviction_order():
    cache = LRUCacheWithTTL(capacity=2)
    cache.put("a", 1)
    cache.put("b", 2)
    
    # Access "a" -> "b" becomes the least recently used item (LRU)
    assert cache.get("a") == 1
    
    # Insert "c" -> evicts "b", retains "a"
    cache.put("c", 3)
    assert cache.get("a") == 1
    assert cache.get("b") is None
    assert cache.get("c") == 3

def test_key_update_does_not_duplicate_order():
    cache = LRUCacheWithTTL(capacity=2)
    cache.put("a", 1)
    cache.put("a", 10)  # Update value for existing key
    cache.put("b", 2)   # Order state: ["a", "b"]
    cache.put("c", 3)   # Must evict "a" without exceeding capacity

    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3
    assert len(cache.cache) <= 2

def test_ttl_expiration_cleanup():
    cache = LRUCacheWithTTL(capacity=3, default_ttl=0.1)
    cache.put("temp", 99, ttl=0.05)
    
    time.sleep(0.08)
    # Key has expired -> must return None and clean up completely
    assert cache.get("temp") is None
    assert "temp" not in cache.cache
    assert "temp" not in cache.expiry
    assert "temp" not in cache.access_order
'''

ADVANCED_TASK_DESC = (
    "Fix the LRUCacheWithTTL implementation: "
    "1. Ensure `get()` updates the key to most-recently-used upon cache hit. "
    "2. Ensure expired keys are cleaned up completely from cache, expiry, and access order upon access. "
    "3. Fix LRU eviction to remove the least recently used item (oldest item in order) when capacity is reached. "
    "4. Ensure updating an existing key updates its position without duplicating keys in the access order list."
)

# ==============================================================================
# 2. UI Custom Styling (Equal height & Alignment CSS)
# ==============================================================================

CUSTOM_CSS = """
<style>
/* Target Streamlit secondary buttons in columns */
div[data-testid="column"] button,
div[data-testid="stButton"] button {
    height: 52px !important;
    min-height: 52px !important;
    max-height: 52px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
}
/* Make Streamlit column containers flex and fill equal height */
div[data-testid="column"] {
    display: flex !important;
    flex-direction: column !important;
}
div[data-testid="column"] > div {
    display: flex !important;
    flex-direction: column !important;
    flex: 1 1 auto !important;
}
/* Target Streamlit text area root, wrapper, and actual textarea */
div[data-testid="stTextArea"] {
    flex: 1 1 auto !important;
    display: flex !important;
    flex-direction: column !important;
}
div[data-testid="stTextArea"] > div:nth-child(2),
div[data-testid="stTextArea"] [data-baseweb="textarea"],
div[data-testid="stTextArea"] textarea {
    height: 480px !important;
    min-height: 480px !important;
    max-height: 480px !important;
    box-sizing: border-box !important;
}
div[data-testid="stTextArea"] textarea {
    font-family: Consolas, "Courier New", monospace !important;
    font-size: 0.88rem !important;
}
/* Harmonize subheader heights across the 3 columns */
div[data-testid="column"] h3,
div[data-testid="column"] [data-testid="stMarkdownContainer"] h3 {
    height: 52px !important;
    min-height: 52px !important;
    max-height: 52px !important;
    display: flex !important;
    align-items: center !important;
    margin: 0 0 8px 0 !important;
    padding: 0 !important;
}
/* Harmonize label heights above text areas */
div[data-testid="stTextArea"] label,
div[data-testid="stWidgetLabel"] {
    height: 28px !important;
    min-height: 28px !important;
    max-height: 28px !important;
    display: flex !important;
    align-items: center !important;
    margin: 0 !important;
    padding: 0 !important;
}
/* Expander styling for Section 1 and inspectors */
div[data-testid="stExpander"] {
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 8px !important;
    margin-bottom: 14px !important;
}
div[data-testid="stExpander"] details summary {
    font-size: 1.08rem !important;
    font-weight: 600 !important;
}
</style>
"""

# ==============================================================================
# 3. Mock LLM Simulator Constants
# ==============================================================================

MOCK_CALCULATOR_FIXED_CODE = '''```python
from typing import List, Union

class Calculator:
    """A mathematical operations calculator with all bugs healed."""

    def add(self, a: float, b: float) -> float:
        return a + b

    def subtract(self, a: float, b: float) -> float:
        return a - b

    def multiply(self, a: float, b: float) -> float:
        return a * b

    def divide(self, a: float, b: float) -> float:
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

    def power(self, base: float, exponent: float) -> float:
        return base ** exponent

    def average(self, numbers: List[Union[int, float]]) -> float:
        if not numbers:
            return 0.0
        return sum(numbers) / len(numbers)

    def factorial(self, n: int) -> int:
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        if n == 0 or n == 1:
            return 1
        return n * self.factorial(n - 1)
```'''


# ==============================================================================
# 4. Agent Prompts
# ==============================================================================

CODE_PATCH_SYSTEM_PROMPT = """You are an expert autonomous Python bug-fixing engine.
Your goal is to inspect the buggy Python source code, review the failing pytest suite and error traceback, and produce the complete corrected Python code.

CRITICAL INSTRUCTIONS:
1. Provide the complete corrected Python code that will make all tests pass.
2. Output the code inside a single ```python ... ``` block or via write(path='solution.py', content='...').
3. Do NOT output conversational text, explanations, or questions.
4. Preserve existing functions and classes, fixing all bugs preventing tests from passing.
5. Pay close attention to assertion failures, cleanup requirements, and edge cases in the test suite.
"""

CODE_PATCH_USER_PROMPT = """TASK DESCRIPTION:
{task_description}

CURRENT CODE (solution.py):
```python
{source_code}
```

TEST SUITE (pytest):
```python
{test_code}
```

{error_section}

Write the complete corrected Python code for solution.py to pass all tests:
"""


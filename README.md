# Lens

A Streamlit app for exploring supermarket sales with pandas and a local Llama
model. The bundled `data/SuperMarket Analysis.csv` contains 1,000 rows and 17
columns and is copied unchanged from the supplied dataset.

## Run

Use Python 3.10 or newer. From this folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull llama3:8b
python -m streamlit run app.py
```

Ollama must be running: open its desktop app or run `ollama serve` in another
terminal. No OpenAI account or API key is needed.

The default model is `llama3:8b`. To use another installed model, set its exact tag
before starting Streamlit:

```powershell
$env:OLLAMA_MODEL = "llama3.1:8b"
```

`OLLAMA_HOST` optionally changes the server address (default
`http://127.0.0.1:11434`). Schema information and calculation results are sent to
that server. With the default address, inference stays on your machine.

## Example questions

- Total Sales by City
- Average Rating by Product line
- Monthly Sales trend by Date
- Correlation between Unit price and Sales
- Outliers in Sales

Upload another UTF-8 CSV through the sidebar. The offline toggle uses a limited
rule-based parser without a model; use exact column names for basic questions.

The memory tab saves definitions, preferences and filters in `memory.sqlite`.
For example, save a filter named `members` with these clauses:

```json
[{"column": "Customer type", "op": "eq", "value": "Member"}]
```

Then ask `Total Sales for members`. Facts are separated by dataset fingerprint.
Saving the same name replaces the previous fact.

## Implementation

`agent.py` contains the pandas calculations and analysis loop. `ollama_client.py`
calls Ollama's `/api/chat` endpoint. Llama returns a JSON calculation request;
the app validates and runs the selected function and returns its result to the
model. It does not execute generated Python. The JSON protocol does not require
native function calling from Llama 3.

Answers include the calculation trace. Models can still misunderstand questions
or describe results incorrectly, so check the evidence. Requests stop after ten
model turns. CSV limits are 10 MB, 100,000 rows and 100 columns. Dates in the
bundled file use month/day/year. Numeric summaries exclude missing values; sums
without valid values return null. Weekly buckets end on Sunday.

## Checks and command line

```powershell
python -m pytest -q
python agent.py "data/SuperMarket Analysis.csv" "Total Sales by City" --offline
python agent.py "data/SuperMarket Analysis.csv" "Total Sales by City"
```

Tests cover calculations, memory, malformed model replies and the Streamlit UI.
The tiny orders fixture under `tests/fixtures` exercises missing values and invalid
dates; it is not an example dataset in the app.

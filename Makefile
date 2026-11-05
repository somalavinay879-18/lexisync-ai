# LexiSyncAI - one-command workflows
PY ?= python3
VENV = .venv
BIN = $(VENV)/bin
CASE ?= GDPR ePrivacy
MODEL ?= qwen2.5:7b

.PHONY: all setup model docs index app eval eval-retrieval eval-resume eval-summary ocr-bench test diagram clean

all: setup model docs index app          ## everything, from zero to the running app

setup:                                   ## virtual environment + pinned dependencies
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements-dev.txt

model:                                   ## download the local LLM (needs Ollama installed)
	ollama pull $(MODEL)

docs:                                    ## download the public PDFs (GDPR, ePrivacy Directive)
	$(BIN)/python scripts/download_docs.py

index:                                   ## parse, chunk and index data/docs/*.pdf into the case
	$(BIN)/python scripts/build_case.py --case "$(CASE)"

app:                                     ## start the web app on http://localhost:8501
	$(BIN)/streamlit run app.py

eval-retrieval:                          ## recall@k for bm25 / dense / hybrid / hybrid+rerank (no LLM)
	$(BIN)/python -m eval.run_eval --case "$(CASE)" --retrieval-only

eval:                                    ## full evaluation incl. LLM answers (takes ~30-60 min)
	$(BIN)/python -m eval.run_eval --case "$(CASE)"

eval-resume:                             ## continue an interrupted `make eval` where it stopped
	$(BIN)/python -m eval.run_eval --case "$(CASE)" --resume

eval-summary:                            ## recompute the summary after manual grading of reports/answers.csv
	$(BIN)/python -m eval.run_eval --case "$(CASE)" --summary-only

ocr-bench:                               ## OCR character accuracy on 5 GDPR pages
	$(BIN)/python scripts/ocr_benchmark.py data/docs/gdpr.pdf --pages 5

test:                                    ## unit tests (offline, no model downloads)
	$(BIN)/python -m pytest -q

diagram:                                 ## render docs/architecture.mmd to PNG
	npx -y @mermaid-js/mermaid-cli -i docs/architecture.mmd -o docs/architecture.png -w 1800 -b white

clean:
	rm -rf data/cases .pytest_cache

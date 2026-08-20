# rag-orc developer tasks.
.DEFAULT_GOAL := help
PY := .venv/bin/python
CLI := .venv/bin/ragorc
UV := uv

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
.PHONY: install
install: ## Create the venv and install with dev extras
	$(UV) venv --python 3.12
	$(UV) pip install -e ".[dev,server,redis,raptor,graphrag,loaders,otel,langchain,web]"

.PHONY: install-all
install-all: ## Install everything, including torch-based extras
	$(UV) pip install -e ".[all,dev,local,nli]"

# ---------------------------------------------------------------------------
.PHONY: up
up: ## Start Postgres, Neo4j and Qdrant
	docker compose up -d
	@echo "waiting for health..."
	@until [ "$$(docker compose ps --format json | grep -c '"Health":"healthy"')" -ge 3 ]; do sleep 2; printf '.'; done; echo " ready"

.PHONY: up-cache
up-cache: ## Start the stack plus Redis
	docker compose --profile cache up -d

.PHONY: down
down: ## Stop the stack (volumes retained)
	docker compose down

.PHONY: nuke
nuke: ## Stop the stack and DELETE all data
	docker compose down -v

.PHONY: doctor
doctor: ## Diagnose "the stack is up but nothing connects"
	@bash scripts/doctor.sh

.PHONY: logs
logs: ## Tail service logs
	docker compose logs -f --tail=100

# ---------------------------------------------------------------------------
.PHONY: lint
lint: ## Ruff check + format check
	$(PY) -m ruff check ragorc tests
	$(PY) -m ruff format --check ragorc tests

.PHONY: fmt
fmt: ## Auto-fix and format
	$(PY) -m ruff check ragorc tests --fix
	$(PY) -m ruff format ragorc tests

.PHONY: types
types: ## mypy
	$(PY) -m mypy ragorc

.PHONY: test
test: ## Unit tests (no services required)
	$(PY) -m pytest tests -m "not integration and not llm and not slow"

.PHONY: test-integration
test-integration: ## Integration tests (requires `make up`)
	$(PY) -m pytest tests -m integration

.PHONY: test-docker
test-docker: ## Integration tests from inside the docker network (no host ports needed)
	docker compose -f docker-compose.yml -f docker/docker-compose.test.yml \
	  run --rm --build integration

.PHONY: test-e2e
test-e2e: ## Real end-to-end query in-network (needs RAGORC_LLM__API_KEY)
	docker compose -f docker-compose.yml -f docker/docker-compose.test.yml \
	  run --rm --build e2e python scripts/e2e_check.py

.PHONY: cov
cov: ## Unit tests with coverage
	$(PY) -m pytest tests -m "not integration and not llm and not slow" \
	  --cov=ragorc --cov-report=term-missing --cov-report=html

.PHONY: check
check: lint test ## Lint + unit tests

# ---------------------------------------------------------------------------
.PHONY: schema
schema: ## Create the Postgres/Qdrant/Neo4j schemas
	$(CLI) init

.PHONY: seed
seed: ## Ingest the example corpus
	$(CLI) ingest examples/corpus --recursive

.PHONY: ask
ask: ## Ask a question: make ask Q="what is late chunking?"
	$(CLI) query "$(Q)"

.PHONY: serve
serve: ## Run the API with reload
	$(PY) -m uvicorn ragorc.server.app:app --reload --port 8000

.PHONY: eval
eval: ## Run the evaluation suite
	$(CLI) eval examples/eval/questions.jsonl

.PHONY: bench
bench: ## Benchmark retrieval strategies
	$(CLI) bench

# ---------------------------------------------------------------------------
.PHONY: diagrams
diagrams: ## Render Mermaid diagrams to SVG (needs mmdc)
	@for f in docs/diagrams/*.mmd; do \
	  echo "rendering $$f"; mmdc -i $$f -o $${f%.mmd}.svg -b transparent; \
	done

.PHONY: clean
clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

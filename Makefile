.PHONY: test test-local seed seed-fake api worker

test:            ## offline CI tier (fake embedder + fake LLM)
	pytest -q

test-local:      ## real bge + real Ollama
	pytest -q -m local_llm

seed:            ## ingest ./fixtures with the real local model, print the catalog
	python -m app.cli seed

seed-fake:       ## same, but with the offline fake embedder (no download)
	python -m app.cli seed --fake

api:             ## serve the HTTP API
	uvicorn app.api:app --reload

worker:          ## run an ingestion worker (run one or more)
	python -m app.worker

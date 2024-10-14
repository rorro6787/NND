.PHONY: lint format

lint:
	@python3 -m ruff check --extend-select I src/backend/

format:
	@python3 -m ruff format src/backend/
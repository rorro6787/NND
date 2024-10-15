.PHONY: lint format

lint:
	@python3 -m ruff check --extend-select I src/backend/

format:
	@python3 -m ruff format src/backend/

fix:
	@python3 -m ruff check src/backend/ --fix
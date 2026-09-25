.PHONY: install lint fmt test run up down logs psql

-include .env
export

install:
	uv sync --locked
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check . --fix
	uv run ruff format .

test:
	uv run pytest --cov

run:
	DATABASE__PORT=5433 \
	DATABASE__USERNAME="$$POSTGRES_USER" \
	DATABASE__PASSWORD="$$POSTGRES_PASSWORD" \
	DATABASE__DATABASE_NAME="$$POSTGRES_DB" \
	uv run uvicorn src.app:app --reload --port 8000

up:
	docker compose up -d --build --wait

down:
	docker compose down

logs:
	docker compose logs -f app

psql:
	docker compose exec postgres sh -c 'psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"'

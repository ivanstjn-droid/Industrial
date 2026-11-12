.PHONY: install test bench serve
install:
	pip install -e ".[dev]"
test:
	pytest
bench:
	anlagenwacht benchmark all
serve:
	anlagenwacht serve

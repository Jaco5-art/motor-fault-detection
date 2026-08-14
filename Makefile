.PHONY: install install-deep test check predict-demo figures

install:
	python -m pip install -e .

install-deep:
	python -m pip install -e ".[deep,viz]"

test:
	python -m unittest discover -s tests -v
	python -m compileall -q src scripts tests

check:
	python scripts/check_pipeline.py --archive "$(ARCHIVE)"

predict-demo:
	motor-fault-predict --archive "$(ARCHIVE)" --run-id "$(RUN_ID)" --archive-split testing

figures:
	python scripts/generate_readme_assets.py

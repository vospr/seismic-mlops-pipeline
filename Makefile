UV ?= uv
RUN = $(UV) run --frozen
F3_DIR ?= data/f3

.PHONY: demo ci test train smoke serve f3-download f3 fixture

demo: ## fresh clone -> install, train on the committed fixture, promote, serve, query
	$(UV) sync --frozen
	rm -f mlflow.db
	$(RUN) python -m seismic_mlops.train --data fixture --fail-if-not-promoted
	$(RUN) python scripts/smoke_serve.py

ci: ## what GitHub Actions runs; any failing step fails the build
	$(UV) sync --frozen
	$(RUN) pytest
	$(MAKE) demo

test:
	$(RUN) pytest

train:
	$(RUN) python -m seismic_mlops.train --data fixture

smoke:
	$(RUN) python scripts/smoke_serve.py

serve:
	MLFLOW_TRACKING_URI=sqlite:///mlflow.db $(RUN) uvicorn --factory seismic_mlops.serve:create_app --port 8000

f3-download: ## full benchmark (1.05 GB), checksum verified
	mkdir -p $(F3_DIR)
	curl -L -C - -o $(F3_DIR)/data.zip https://zenodo.org/record/3755060/files/data.zip
	$(RUN) python -c "from pathlib import Path; from seismic_mlops.data import md5_matches; import sys; sys.exit(0 if md5_matches(Path('$(F3_DIR)/data.zip')) else 'MD5 mismatch')"
	unzip -q -o $(F3_DIR)/data.zip -d $(F3_DIR)

f3: ## train + gate on the real benchmark
	$(RUN) python -m seismic_mlops.train --data $(F3_DIR)/data

fixture: ## regenerate the committed fixture from the downloaded benchmark
	$(RUN) python -m seismic_mlops.data fixture $(F3_DIR)/data

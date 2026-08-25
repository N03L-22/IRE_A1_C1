# A1 Component-1 -- one-command rebuild (Q1.5).
#
#   make data       extract the working tier
#   make smoke      walking skeleton on EB-NeRD demo (fast sanity check)
#   make test       run the test suite
#   make clean      remove derived data, keep the raw archives
#
# Resource budget (see architecture.md 7b): override per invocation, e.g.
#   make smoke N_JOBS=8 MEM_GB=8

PYTHON  := .venv/bin/python
RAW     := data/raw
WORK    := data/work
N_JOBS  ?= 28
MEM_GB  ?= 28
BUDGET  := --n-jobs $(N_JOBS) --mem-gb $(MEM_GB)

.PHONY: all data data-demo data-artifacts store smoke bm25 bm25-sweep eval test clean clean-all help

all: store test

## data: extract the small tier -- the working and headline tier
data: $(WORK)/.small.stamp

$(WORK)/.small.stamp: $(RAW)/mind/MINDsmall_train.zip \
                      $(RAW)/mind/MINDsmall_dev.zip \
                      $(RAW)/ebnerd/ebnerd_small.zip
	$(PYTHON) -m src.data.extract --tier small
	@mkdir -p $(WORK) && touch $@

## data-demo: extract EB-NeRD demo -- the smoke-test tier
data-demo: $(WORK)/.demo.stamp

$(WORK)/.demo.stamp: $(RAW)/ebnerd/ebnerd_demo.zip
	$(PYTHON) -m src.data.extract --tier demo
	@mkdir -p $(WORK) && touch $@

## data-artifacts: extract the provided EB-NeRD embeddings (Phase 3 baseline)
data-artifacts:
	$(PYTHON) -m src.data.extract --tier artifacts

## store: unify + temporally split both datasets -> data/store (Q1.2-Q1.4)
store: data
	$(PYTHON) -m src.data.clean --tier small $(BUDGET)

## smoke: walking skeleton end to end on the demo tier
smoke: data-demo
	$(PYTHON) -m src.skeleton --dataset ebnerd --tier demo --limit 200 $(BUDGET)

## bm25: BM25 vs baselines on both datasets (Q2, exploratory -- no CIs)
bm25: store
	$(PYTHON) -m src.retrieval.run_bm25 --dataset ebnerd $(BUDGET) \
	    --out results/bm25_ebnerd.json
	$(PYTHON) -m src.retrieval.run_bm25 --dataset mind $(BUDGET) \
	    --out results/bm25_mind.json

## bm25-sweep: k1/b/last_n/window sweep on val only, parallel (Q2 D5)
bm25-sweep: store
	$(PYTHON) -m src.retrieval.run_bm25 --dataset ebnerd --sweep --limit 4000 \
	    $(BUDGET) --out results/bm25_sweep_ebnerd.json
	$(PYTHON) -m src.retrieval.run_bm25 --dataset mind --sweep --limit 4000 \
	    $(BUDGET) --out results/bm25_sweep_mind.json

## eval: score every retriever through the one harness, with CIs (Q4)
eval: store
	$(PYTHON) -m src.eval.run --dataset ebnerd --limit 4000 $(BUDGET) \
	    --out results/eval_ebnerd.json
	$(PYTHON) -m src.eval.run --dataset mind --limit 4000 $(BUDGET) \
	    --out results/eval_mind.json

## test: run the test suite
test:
	$(PYTHON) -m pytest tests/ -q

## clean: remove extracted/derived data, keep the raw archives
clean:
	rm -rf $(WORK) data/store
	@echo "removed $(WORK) and data/store; $(RAW) untouched"

## clean-all: also remove the venv
clean-all: clean
	rm -rf .venv

help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## /  /'

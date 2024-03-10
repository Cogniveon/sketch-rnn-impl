GUM ?= tools/gum

.PHONY: help run
.DEFAULT_GOAL := run

help: ## Makefile documentation
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

SELECTED_OPTION ?= $(MAKE) --no-print-directory help | $(GUM) choose | awk '{print $$1}'
run: ## Interactive run menu for Make commands
	@$(SELECTED_OPTION) | xargs $(MAKE) --no-print-directory


clean_dataset: ## Deletes the /data directory
	@rm -rf data/

DOWNLOAD_URL := "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"
API_URL := "https://storage.googleapis.com/storage/v1/b/quickdraw_dataset/o?prefix=full/numpy_bitmap/&fields=items(name)"
download_dataset: ## Download a dataset from the Quickdraw dataset (optionally specify DATASET=rabbit)
	@if [ -z "$(DATASET)" ]; then \
			echo "Fetching available datasets..."; \
			datasets=$$(curl -s $(API_URL) |  jq -r '.items[] | .name | select(. | endswith(".npy")) | rtrimstr(".npy") | split("/") | .[-1]' | sed 's/.*/"&"/' | sed 's/ /_/g' | tr '\n' ' '); \
			if [ -z "$$datasets" ]; then \
					echo "No datasets found. Exiting."; \
					exit 1; \
			fi; \
			echo "Select a dataset to download:"; \
			choice=$$(echo $$datasets | tr ' ' '\n' | $(GUM) filter --limit 1 | xargs echo); \
			if [ -z "$$choice" ]; then \
					echo "No dataset selected. Exiting."; \
					exit 1; \
			fi; \
	else \
			choice=$(DATASET); \
	fi; \
	if [ -z "$$choice" ]; then \
			echo "No dataset selected. Exiting."; \
			exit 1; \
	fi; \
	mkdir -p data/quickdraw; \
	urlencoded_choice=$$(echo -n $$choice | sed 's/_/%20/g' | xargs echo); \
	curl -o data/quickdraw/$$choice.npy $(DOWNLOAD_URL)/$$urlencoded_choice.npy; \
	echo "$$choice.npy has been downloaded to data/quickdraw/"
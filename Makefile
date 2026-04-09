.PHONY: clean
clean: ## Remove build artifacts and caches
    rm -rf build/ dist/ *.egg-info .eggs/
    rm -rf aws_infra/*.egg-info aws_infra/build/ aws_infra/dist/
    rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
    rm -rf htmlcov/ .coverage coverage.xml
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name '*.pyc' -delete 2>/dev/null || true

.PHONY: redeploy
redeploy:
	docker build -t something:latest . && docker run --rm -v $(pwd):/app -w /app something:latest


docker build -t something:latest . 
docker run --rm -p 8000:8000 somename:latest


python -m cloud_server.app --reload
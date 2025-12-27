# Makefile for Titanic Survival Prediction
# Convenience commands for common tasks

.PHONY: install train train-advanced predict clean test help

# Install dependencies
install:
	pip install -r requirements.txt

# Train basic model
train:
	python src/train_model.py

# Train advanced models with hyperparameter tuning
train-advanced:
	python src/advanced_train.py

# Run prediction demo
predict:
	python src/predict.py

# Clean generated files
clean:
	rm -rf __pycache__ src/__pycache__
	rm -rf .pytest_cache .mypy_cache

# Show available commands
help:
	@echo "Available commands:"
	@echo "  make install        - Install dependencies"
	@echo "  make train          - Train basic model"
	@echo "  make train-advanced - Train advanced models"
	@echo "  make predict        - Run prediction demo"
	@echo "  make clean          - Clean generated files"


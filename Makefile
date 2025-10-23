SHELL := /bin/bash

CITY ?= nyc

venv:
	python -m venv .venv && source .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

features:
	python -m src.features.build_features --city $(CITY)

train:
	python -m src.models.train_price_model --city $(CITY)

app:
	streamlit run app/streamlit_app.py

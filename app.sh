#!/bin/bash
uvicorn api.main:app --host 0.0.0.0 --port 7860 --reload

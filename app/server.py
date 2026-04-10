from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="WiFi Fall Alert Dashboard")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

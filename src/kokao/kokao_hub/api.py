"""Kokao Hub API — Mock для тестов."""

from fastapi import FastAPI, HTTPException, status

app = FastAPI(title="Kokao Engine API")


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/engines")
def list_engines():
    """List available engines."""
    return {"engines": [], "total": 0}


@app.get("/engines/search")
def search_engines(q: str = ""):
    """Search engines by query."""
    return {"results": [], "query": q}


@app.get("/download/{engine_id}")
def download_engine(engine_id: str):
    """Download engine by ID."""
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, detail=f"Engine {engine_id} not found"
    )


@app.get("/docs")
def docs():
    """Documentation URL."""
    return {"docs_url": "/docs", "openapi_url": "/openapi.json"}


@app.get("/openapi.json")
def openapi():
    """OpenAPI schema."""
    return app.openapi()

"""CLI entry points for Superbrain.

Uses typer for command parsing. All operations call the running API via HTTP
so the server must be up. The API base URL defaults to http://localhost:8000
and can be overridden with SUPERBRAIN_API_BASE_URL.

Usage:
    superbrain ingest url <url>
    superbrain ingest status <job_id>
    superbrain health
"""


import httpx
import typer

from superbrain.settings import get_settings

app = typer.Typer(help="Superbrain command-line interface")
ingest_app = typer.Typer(help="Ingestion job commands")
digest_app = typer.Typer(help="Digest commands")
app.add_typer(ingest_app, name="ingest")
app.add_typer(digest_app, name="digest")


def _base_url() -> str:
    """Return the API base URL from settings.

    Returns:
        The base URL string without a trailing slash.
    """
    return get_settings().api_base_url.rstrip("/")


@ingest_app.command("url")
def ingest_url(url: str = typer.Argument(..., help="https URL to ingest")) -> None:
    """Create an ingestion job for a URL.

    Args:
        url: The https URL to submit for ingestion.
    """
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                f"{_base_url()}/ingestion/jobs",
                json={"input_type": "url", "input_value": url},
            )
        if resp.status_code == 422 or resp.status_code == 400:
            data = resp.json()
            typer.echo(f"Error: {data.get('detail') or data.get('message')}", err=True)
            raise typer.Exit(1)
        resp.raise_for_status()
        job = resp.json()
        typer.echo(f"Job created: {job['id']}")
    except httpx.ConnectError:
        typer.echo(f"Cannot connect to API at {_base_url()}. Is the server running?", err=True)
        raise typer.Exit(1)


@ingest_app.command("status")
def ingest_status(job_id: str = typer.Argument(..., help="Job UUID to check")) -> None:
    """Poll the status of an ingestion job.

    Args:
        job_id: The UUID of the job to check.
    """
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{_base_url()}/ingestion/jobs/{job_id}")
        if resp.status_code == 404:
            typer.echo(f"Job {job_id} not found.", err=True)
            raise typer.Exit(1)
        resp.raise_for_status()
        job = resp.json()
        typer.echo(f"Job {job['id']}: {job['status']}")
        if job.get("error_message"):
            typer.echo(f"  Error: {job['error_message']}")
    except httpx.ConnectError:
        typer.echo(f"Cannot connect to API at {_base_url()}. Is the server running?", err=True)
        raise typer.Exit(1)


@digest_app.command("trigger")
def digest_trigger(
    date: str | None = typer.Option(None, help="Date to digest (YYYY-MM-DD). Defaults to yesterday."),
) -> None:
    """Trigger a daily digest generation run via the API."""
    try:
        payload: dict = {}
        if date:
            payload["date"] = date
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{_base_url()}/digests/trigger", json=payload)
        resp.raise_for_status()
        data = resp.json()
        typer.echo(f"Digest queued: {data.get('detail')} (date: {data.get('date')})")
    except httpx.ConnectError:
        typer.echo(f"Cannot connect to API at {_base_url()}. Is the server running?", err=True)
        raise typer.Exit(1)


@app.command()
def health() -> None:
    """Check the health of the Superbrain API and its dependencies."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{_base_url()}/health")
        resp.raise_for_status()
        data = resp.json()
        typer.echo(f"status:  {data['status']}")
        typer.echo(f"db:      {data['db']}")
        typer.echo(f"ollama:  {data['ollama']}")
    except httpx.ConnectError:
        typer.echo(f"Cannot connect to API at {_base_url()}. Is the server running?", err=True)
        raise typer.Exit(1)

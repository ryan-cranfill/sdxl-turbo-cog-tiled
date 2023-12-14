import os
import sys
import httpx
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.logger import logger
from pydantic import BaseSettings
from fastapi.middleware.cors import CORSMiddleware


class Settings(BaseSettings):
    BASE_URL = "http://localhost:8000"
    USE_NGROK = os.environ.get("USE_NGROK", "False") == "True"


settings = Settings()


# def init_webhooks(base_url):
#     # Update inbound traffic via APIs to use the public-facing ngrok URL
#     pass


# Initialize the FastAPI app for a simple web server
app = FastAPI()

if settings.USE_NGROK:
    # pyngrok should only ever be installed or initialized in a dev environment when this flag is set
    from pyngrok import ngrok

    # Get the dev server port (defaults to 8000 for Uvicorn, can be overridden with `--port`
    # when starting the server
    port = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else "8000"

    # Open a ngrok tunnel to the dev server
    public_url = ngrok.connect(port).public_url
    logger.info("ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}\"".format(public_url, port))

    # Update any base URLs or webhooks to use the public ngrok URL
    settings.BASE_URL = public_url
    # init_webhooks(public_url)

# Add CORs middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Just a dumb proxy to pass along the request to the model server
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy(path: str, request: Request):
    base_url = "http://localhost:5001"
    async with httpx.AsyncClient() as client:
        if request.method == "GET":
            resp = await client.get(f"{base_url}/{path}", params=request.query_params)
        elif request.method == "POST":
            resp = await client.post(f"{base_url}/{path}", data=await request.body())
        elif request.method == "PUT":
            resp = await client.put(f"{base_url}/{path}", data=await request.body())
        elif request.method == "DELETE":
            resp = await client.delete(f"{base_url}/{path}")
        elif request.method == "PATCH":
            resp = await client.patch(f"{base_url}/{path}", data=await request.body())
        elif request.method == "OPTIONS":
            resp = await client.options(f"{base_url}/{path}")
        else:
            raise HTTPException(status_code=405, detail="Method Not Allowed")

    return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("fastapi_proxy:app", host="0.0.0.0", port=8000)

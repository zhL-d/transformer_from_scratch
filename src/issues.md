1. current Dockerfile creates .venv inside /app, but you bind-mount /app, which hides that venv. Pick one (both are fine):

2. there are too many config layer, github action, start script, yml file, code



Infra shape: Bicep + environment param files (dev/test/prod).
Runtime config: env vars injected at run time (ACA Job, or a single azure.env file consumed by the workflow).
Local dev: .env for Compose only.
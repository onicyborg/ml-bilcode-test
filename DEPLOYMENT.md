# Deployment

Push/merge into development builds an immutable GHCR image and deploys over SSH. main is the primary branch; merging main alone does not deploy. Configure the Development GitHub environment before triggering a deployment.

Secrets: VPS_HOST, VPS_USER, VPS_SSH_KEY, optional VPS_SSH_PASSPHRASE, VPS_KNOWN_HOSTS (trusted host key verified out of band), VPS_ENV (complete runtime .env). GITHUB_TOKEN is automatic. Variable VPS_APP_DIR defaults to /opt/siklus/ml; use an absolute path without spaces. The SSH user needs Docker and permission to create that directory. VPS needs Docker Compose v2. Images target the runner's amd64 architecture.

Deploy ML first, then backend, then frontend. Shared Docker network is siklus. Set backend ML_URL=http://ml:8000; frontend BACKEND_URL=http://api:3001 and WEB_ORIGIN to the public HTTPS website URL. Backend CORS_ORIGINS should list the website origin. Set PORT=8000. Use a reverse proxy such as Caddy for public HTTPS; container host ports bind loopback only. Do not expose ML publicly. Configure Neon PostgreSQL 18 DATABASE_URL and verify uuidv7(). Backend deployment runs prisma migrate deploy before replacing the running API; never run migrate dev in production. Do not auto-seed production accounts.

The runtime env file and temporary registry login are not baked into images. Protect environment secrets and branch deployment permissions. Failed readiness fails the workflow; automatic rollback is not implemented. Roll back by redeploying a previously verified SHA image. Run one API instance so the in-process daily purge scheduler does not duplicate work.

For frontend hosted on Vercel instead, use its Git integration and configure development as the deployment branch; disable the VPS workflow if it is not used. The Next.js proxy still has Vercel upload-size limits that require a separate upload-flow adjustment before full 5 MiB support.

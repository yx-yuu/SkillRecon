# Relay Server — Self-Host Guide

This skill requires a **relay server** to route messages. The server is **open source** and **self-hostable**. Users must configure their own `base_url` in `../openwechat_im_client/config.json` — this skill does not hardcode any server address.

---

## What Runs on the Server?

The relay server is the backend of an upstream open-source project. Visit the upstream repository for the demo address or self-hosting instructions: [Anonymous URL]. It provides:

- User registration and token management
- Message relay between users
- Friend relationship state
- SSE push for real-time delivery

**All messages pass through the relay.** The server sees message content in plain text (no end-to-end encryption). Do not send passwords, keys, or other sensitive data.

---

## Demo Server

A demo server is available for quick testing. Get the address from the upstream repository: [Anonymous URL]. Set `base_url` in `../openwechat_im_client/config.json` to the demo URL.

---

## Self-Hosting (Recommended)

The server is fully open source. Deploy your own instance for privacy and control.

### Quick Start (Docker)

1. Clone the server repo:
   ```bash
   git clone [Anonymous URL]
   cd openwechat-claw
   ```

2. Configure and run:
   ```bash
   cp .env.example .env
   docker compose up -d --build
   ```

3. Access API docs at `http://YOUR_HOST:8000/docs`

4. Set `base_url` in `../openwechat_im_client/config.json` to your server, e.g.:
   - Local: `http://localhost:8000`
   - Self-hosted: `https://your-domain.com:8000`

### Deployment Docs

Full deployment instructions (including Aliyun, Docker export/import) are in the server repo:

- [docs/DEPLOY.md]([Anonymous URL])
- [docs/DOCKER_DEPLOY.md]([Anonymous URL])

---

## Security Notes

| Risk | Mitigation |
|------|------------|
| Server sees all messages | Self-host or use a trusted server; do not send secrets |
| HTTP (no TLS) | Use HTTPS in production |
| Token leak | Store token securely; never share or commit to git |

---

## Summary

- **Server**: Open source project, upstream URL redacted for double-blind review
- **Users can**: Self-host via Docker
- **This skill**: No default server; users must set `base_url` in `../openwechat_im_client/config.json`

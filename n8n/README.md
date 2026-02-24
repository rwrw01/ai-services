# n8n — Orchestration Service

Runs n8n for scheduling and orchestrating the daily news briefing pipeline.

## Setup

1. Start the service:
   ```bash
   cd /home/ralph/services && docker compose up -d n8n
   ```

2. Open the n8n UI at **http://localhost:5678**

3. Log in with credentials from `services/.env` (`N8N_USER` / `N8N_PASSWORD`)

4. Import the workflow:
   - Go to **Workflows** → **Import from File**
   - Select `services/n8n/workflows/news-briefing.json`
   - Click **Save** → **Activate** (toggle top-right)

## Workflow: News Briefing — Daily Pipeline

**Schedule:** 04:30 daily (cron)

### Pipeline

```
04:30  Cron trigger → Fetch RSS (NOS, NU.nl, Tweakers)
       → Merge → Deduplicate by URL → Limit 20
       → Per article:
           1. POST backend:8000/api/news/ingest/article         (create)
           2. POST tts:8002/api/tts/synthesize engine=parkiet   (title + description → MP3)
           3. POST backend:8000/.../audio?engine=parkiet        (upload)
```

> **Note:** No LLM summarization — RSS description is used directly as TTS text.

### Manual Trigger

The workflow also has a webhook at `/webhook/news-refresh` for manual refresh
from the PWA. The app calls `POST /api/news/refresh` which proxies to this webhook.

### Timeouts

- Parkiet TTS: 300s per article (GPU-intensive)

### RSS Feeds

Default feeds (configurable in the workflow):
- NOS: `https://feeds.nos.nl/nosnieuwsalgemeen`
- NU.nl: `https://www.nu.nl/rss/Algemeen`
- Tweakers: `https://feeds.tweakers.net/mixed.xml`

## Troubleshooting

- **n8n can't reach backend:** Verify both are on `ai-net` network: `docker network inspect ai-net`
- **TTS timeout:** Parkiet can take 2-4 minutes per article. Increase timeout in the HTTP Request node.
- **RSS empty:** Some feeds may be rate-limited. Check n8n execution logs for HTTP errors.

# Hosted demo on AWS (how https://newinegpt.ashinity.com runs)

One Ubuntu 24.04 arm64 server (EC2 **c8g.2xlarge**, 8 Graviton4 cores, 16 GB, eu-west-1) running the same app as the desktop
download, in public mode, behind nginx with a Let's Encrypt certificate. No graphics card: on 8 Graviton4 cores a new long question
takes about 45 s and a short one about 6 s (14 Nov 2026; on a t4g.xlarge the same questions took about 110 s and 18 s).

## Files here

| File | Goes to |
|---|---|
| `newinegpt.service` | `/etc/systemd/system/newinegpt.service` (runs as user `newinegpt`, public mode, port 7860) |
| `newinegpt.service.d-threads.conf` | `/etc/systemd/system/newinegpt.service.d/threads.conf` (use every core of whatever size the server is) |
| `nginx-newinegpt.conf` | `/etc/nginx/sites-available/newinegpt`, linked in `sites-enabled` (http to https, proxy with 900 s timeouts) |

## Setup in short

1. Security group: 80 and 443 open; 22 only from the admin's address.
2. `apt install nginx python3-numpy certbot python3-certbot-nginx`; user `newinegpt` with home `/opt/newinegpt`.
3. `/opt/newinegpt/llama/`: llama.cpp b9580 `bin-ubuntu-arm64` release (checksum verified).
4. `/opt/newinegpt/app/`: this repository's `app/`, `train/prompt.py`, plus from the GitHub release the archive and search model (inside
   the Linux zip: `pack/`, `models/`) and both Q5 model parts in `dist/models/newinegpt-4b-v6/` (checksums verified).
5. Install the files above, `systemctl enable --now newinegpt`, reload nginx.
6. DNS: an `A` record for the name, **DNS only** if it is on Cloudflare (Cloudflare's proxy cuts off replies slower than 100 s).
7. `certbot certonly --nginx -d <name>`; the certbot timer renews it.

## What the demo keeps

Public mode (`NEWINEGPT_PUBLIC=1`) keeps, without names or addresses: every answer record in
`/opt/newinegpt/app/eval/human_use/public_answers.jsonl` (question, route, passages, raw model output, final answer, checks, time) under a
random id, and every Good/Partly/Wrong click in `public_ratings.jsonl` with that id. Visitors are told so under the question box; their
conversation list stays in their own browser. nginx's standard access log (IP addresses, 14 days) is separate.

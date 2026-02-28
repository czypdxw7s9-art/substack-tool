# Substack Export + Summary Script

Script: `substack_exporter.py`

## What it does
- Reads post URLs from Substack feed (`/feed`), archive API pagination (`/api/v1/archive`), and archive pages (`/archive`)
- Fetches each post page
- Extracts:
  - title
  - URL
  - publication date (`YYYY-MM-DD`)
  - theme label (`politics`, `economics`, `autobiography`, `history`, `science`, `psychology`, `religion`)
  - `subthemes` (one assigned one-word subtheme per URL, chosen from a 3-5 word pool per theme)
  - like count (best effort)
  - comment count (best effort)
  - restack count (best effort)
  - article/main/footnote word counts (comments + auxiliary widgets excluded; footnotes from `.footnote-content`)
  - `image_count` (main body `.captioned-image-container` images)
- Optionally generates an AI summary for each post via OpenAI Responses API
  - `summary_short_neutral`: short (~10 words, max 15), neutral tone
  - `summary_short_mimic`: short (~10 words, max 15), mimics article tone
  - `summary_very_short_snappy`: very short (~5 words, max 10), snappy
  - `summary_dl`: strict single-sentence 10-word summary (no lists/clauses/fragments)
  - `best_quote`: 1-3 sentence direct quote from article text
  - `best_quote_verified`: `yes/no` exact-ish match check against extracted article text
  - `summary_long`: longer (~50 words)
- Appends rows to a CSV file as it runs

## Install deps
```bash
python3 -m pip install requests beautifulsoup4
```

## Quick test (your sample target)
```bash
python3 substack_exporter.py \
  --substack-url https://50wdj4945.substack.com \
  --output-csv 50wdj4945_posts.csv
```

## Single post mode (no feed crawl)
```bash
python3 substack_exporter.py \
  --post-url https://example.substack.com/p/some-post \
  --output-csv single_post.csv
```

## Reprocess URLs from an existing CSV
```bash
python3 substack_exporter.py \
  --urls-from-csv existing.csv \
  --output-csv refreshed.csv
```

To preserve existing summary columns while recalculating counts/word counts:
```bash
python3 substack_exporter.py \
  --urls-from-csv existing.csv \
  --merge-summaries-from-csv existing.csv \
  --no-summaries \
  --output-csv refreshed.csv
```

## Update Existing CSV In Place (By URL)
Overwrite only selected columns while leaving all others unchanged:
```bash
python3 substack_exporter.py \
  --update-existing-csv existing.csv \
  --update-columns best_quote,best_quote_verified \
  --summary-variants best_quote \
  --no-theme
```

Optional row range (1-based, inclusive), e.g. only rows 1-20:
```bash
python3 substack_exporter.py \
  --update-existing-csv existing.csv \
  --update-columns best_quote,best_quote_verified \
  --update-row-range 1-20 \
  --summary-variants best_quote \
  --no-theme
```

## Local corpus cache (recommended)
Download article HTML once, then reuse it for future runs:
```bash
python3 substack_exporter.py \
  --urls-from-csv existing.csv \
  --cache-dir article_cache \
  --output-csv first_pass.csv
```

Reuse local cache for summary experiments (minimal Substack traffic):
```bash
python3 substack_exporter.py \
  --urls-from-csv existing.csv \
  --cache-dir article_cache \
  --output-csv second_pass.csv
```

Force cache refresh:
```bash
python3 substack_exporter.py \
  --urls-from-csv existing.csv \
  --cache-dir article_cache \
  --refresh-cache \
  --output-csv refreshed.csv
```

## Internal link network graph
Build a directed graph of internal post-to-post links and combine repeated anchor text labels per edge.
- Node size scales by inbound links.
- Node color maps publication date from oldest (orange) to newest (blue) across the HSV hue path (including yellow/green when present).
- Sidebar shows selected article date plus incoming/outgoing links with anchor-context snippets (about 10 words on each side).
- Sidebar is a floating panel that appears only after node click.

Graph-only mode from cached pages:
```bash
python3 substack_exporter.py \
  --urls-from-csv existing.csv \
  --cache-dir article_cache \
  --link-graph-only \
  --link-graph-html internal_links_graph.html \
  --link-graph-json internal_links_graph.json \
  --link-graph-dot internal_links_graph.dot
```

You can also generate graph outputs during a normal export run by adding:
- `--link-graph-html <path>`
- `--link-graph-html-standalone` to embed JS for offline one-file sharing
- `--link-graph-dot <path>`
- `--link-graph-json <path>`

Re-render visualization without recrawling/reprocessing posts:
```bash
python3 substack_exporter.py \
  --render-link-graph-from-json internal_links_graph.json \
  --link-graph-html-standalone \
  --link-graph-html internal_links_graph_v2.html
```

Tip: use `--urls-from-csv` when building JSON so publication dates from the CSV flow into node colors/sidebar.

If you only have DOT:
```bash
python3 substack_exporter.py \
  --render-link-graph-from-dot internal_links_graph.dot \
  --link-graph-html internal_links_graph_v2.html
```

## Enable summaries
Set API key:
```bash
export OPENAI_API_KEY="<your key>"
```

Then run:
```bash
python3 substack_exporter.py \
  --substack-url https://50wdj4945.substack.com \
  --output-csv 50wdj4945_posts.csv \
  --openai-model gpt-4.1-mini
```

## For paywalled Substacks (later)
You can pass authenticated cookies either as:
- raw header: `--cookie-header 'a=...; b=...'`
- JSON file: `--cookies-json cookies.json` (supports cookie list or Playwright `storage_state.json`)

Cookies JSON format:
```json
[
  {"name":"substack.sid","value":"...","domain":".substack.com","path":"/"}
]
```

Playwright storage state format is also accepted:
```json
{"cookies":[{"name":"substack.sid","value":"...","domain":".substack.com","path":"/"}]}
```

### Auth test for your Deepleft post
1. Log into Substack in your browser.
2. Copy a valid Cookie header for `deepleft.substack.com` or `substack.com`.
3. Run:
```bash
python3 substack_exporter.py \
  --post-url https://deepleft.substack.com/p/the-full-story-of-meeting-wbe \
  --cookie-header 'substack.sid=...; substack.app_session=...' \
  --require-full-text \
  --output-csv deepleft_auth_test.csv
```

If auth fails, row status will be `auth_error`.

## Useful flags
- `--max-posts 10` to limit scope while testing
  - if `--max-posts` exceeds feed size, script continues into archive pages
- `--no-summaries` to skip AI calls
- `--no-theme` to skip LLM theme classification
- `--no-subthemes` to skip LLM subtheme generation
- `--resume` to skip only URLs with prior `status=ok` (retries previous errors)
- `--sleep 0.5` to slow down requests
- `--summary-sleep 2.0` to reduce summary request burst rate
- `--summary-max-retries 8` to retry on 429/5xx
- `--summary-backoff-base 3.0` for longer exponential backoff
- `--summary-variants none` to skip summary variants while still allowing theme/subtheme generation
- `--post-url <url>` to process one specific post directly
- `--require-full-text` to fail row when paywall markers are detected
- `--link-graph-html <path>` to write an interactive directed graph HTML
- `--link-graph-html-standalone` to embed vis-network JS in the HTML (offline shareable)
- `--link-graph-dot <path>` to write a Graphviz DOT graph
- `--link-graph-json <path>` to write graph data for later re-rendering
- `--link-graph-only` to skip CSV/summaries and only build graph outputs
- `--render-link-graph-from-json <path>` to render graph output(s) from saved JSON without crawling posts
- `--render-link-graph-from-dot <path>` to render graph output(s) from saved DOT without crawling posts

## CSV columns
`title,url,publication_date,likes,comments,restacks,article_word_count,main_word_count,footnote_word_count,image_count,theme,subthemes,summary_short_neutral,summary_short_mimic,summary_very_short_snappy,summary_dl,best_quote,best_quote_verified,summary_long,status,error`

`status` values:
- `ok`
- `auth_error`
- `summary_error`
- `fetch_error`

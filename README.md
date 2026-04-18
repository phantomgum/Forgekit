# TrendGenie

AI-powered project ideas built from what's actually trending in tech right now.

---

## What it does

TrendGenie has two modes:

**Discovery** — Scrapes GitHub trending and Hacker News in real time, feeds that context to Claude, and generates three creative project ideas based on what's genuinely hot right now. Refresh anytime to get a fresh set grounded in the current moment.

**Intent** — Tell it what technology or tool you want to learn. It generates one scoped, creative, portfolio-worthy project idea designed to force deep hands-on learning with that specific technology.

Both modes stream responses character by character so ideas appear as they're generated.

---

## Why I built this

I kept seeing people on the internet building impressive things with the latest tools and feeling behind. Reading newsletters and following the news wasn't enough — I needed something that bridged the gap between knowing about a technology and actually building with it.

TrendGenie is that bridge. It turns trending tech into concrete things to build.

---

## Tech stack

- **Backend** — Python, FastAPI, Server-Sent Events for streaming
- **Frontend** — Vanilla HTML + JavaScript
- **AI** — Anthropic API (Claude), raw with no wrappers
- **Retrieval** — GitHub trending scraper (BeautifulSoup), Hacker News API
- **Streaming** — Custom delimiter-based streaming format with character queue typewriter rendering

---

## How it works

### Discovery mode
```
GitHub trending + HackerNews top stories
→ Scraped and formatted into context
→ Injected into Claude prompt
→ Streamed back as 3 project ideas
→ Rendered card by card with typewriter effect
```

### Intent mode
```
User inputs a technology they want to learn
→ Sent to Claude with a structured system prompt
→ Streamed back as one scoped project idea
→ Rendered with typewriter effect
```

This is practical RAG — no vector database, just real-time retrieval of what's trending injected directly into the prompt as context.

---

## Getting started

### Prerequisites
- Python 3.9+
- Anthropic API key

### Installation

```bash
git clone https://github.com/phantomgum/TrendGenie
cd TrendGenie
pip install -r requirements.txt
```

### Set your API key

```bash
export ANTHROPIC_API_KEY=your_key_here
```

### Run

```bash
uvicorn main:app --reload
```

Open `index.html` in your browser or serve it locally.

---

## Project structure

```
TrendGenie/
├── main.py          # FastAPI server, endpoints, Anthropic API calls
├── index.html       # Frontend — single file, vanilla JS
├── requirements.txt
└── README.md
```

---

## What I learned building this

- How to call the Anthropic API directly without wrappers and handle streaming responses
- How to implement Server-Sent Events in FastAPI for real-time streaming
- How retrieval-augmented generation works in practice — scraping real context and injecting it into prompts
- How to design a custom streaming format for progressive UI rendering
- Why prompt design is the most important layer in any AI-powered app

---

## Roadmap / Future Plans

- [ ] Saved ideas — persist ideas you want to come back to
- [ ] TLDR AI feed — add newsletter sources to discovery retrieval
- [ ] Difficulty filter — scope ideas by time available
- [ ] More data sources — Product Hunt, dev.to, arXiv

---

## License

MIT

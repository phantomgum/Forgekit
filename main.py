import os
import json
from dotenv import load_dotenv

load_dotenv()
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, Union, List
from bs4 import BeautifulSoup
from supabase import create_client, Client

app = FastAPI()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

class SavedIdea(BaseModel):
    mode: str
    title: str
    tagline: str
    description: str
    inspired_by: Optional[str] = ""
    what_youll_learn: Union[str, List[str]]
    tools_and_tech: Union[str, List[str]]
    first_step: str
    estimated_time: str

def stream_anthropic(prompt: str, max_tokens: int):
    if not ANTHROPIC_API_KEY:
        yield f"data: {json.dumps('ANTHROPIC_API_KEY not set')}\n\n"
        yield "data: [DONE]\n\n"
        return

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    payload = { 
        "model": "claude-sonnet-4-6",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True
    }

    try:
        with requests.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers, stream=True) as response:
            if response.status_code != 200:
                yield f"data: {json.dumps(f'Error: {response.text}')}\n\n"
                yield "data: [DONE]\n\n"
                return

            for line in response.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith('data:'):
                        data_str = decoded[5:].strip()
                        try:
                            event_data = json.loads(data_str)
                            if event_data.get("type") == "content_block_delta":
                                text = event_data.get("delta", {}).get("text", "")
                                if text:
                                    yield f"data: {json.dumps(text)}\n\n"
                        except json.JSONDecodeError:
                            pass
            yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps(f'Error: {str(e)}')}\n\n"
        yield "data: [DONE]\n\n"

@app.get("/generate-idea")
def generate_idea(goal: str = Query(...)):

    prompt = f"""
    You are an expert software engineer and creative project mentor. Your job is to design impressive, creative, and genuinely useful project ideas that force the developer to deeply learn a specific technology by building something real with it.

You are given a technology or tool the user wants to learn :{goal}. You generate ONE project idea that meets all of these criteria:

- CREATIVE: Not a todo app, not a clone, not a tutorial project. Something that would make someone say "that's actually a cool idea"
- SCOPED: Completable in a weekend to two weeks solo. Not a startup, not a platform. One focused thing.
- FORCED LEARNING: The technology must be central to the project — you cannot build it without deeply using that technology. It cannot be an afterthought.
- REAL UTILITY: It solves an actual problem or does something genuinely interesting. Not just a demo.
- IMPRESSIVE: The kind of project that looks good in a GitHub portfolio and sparks conversation.

Return your response using ONLY the following exact format, field by field, with no other text before or after, wrapping the actual text in these exact simple delimiters:

[TITLE]Short punchy project name[/TITLE]
[TAGLINE]One sentence that makes someone want to build it[/TAGLINE]
[DESCRIPTION]2-3 sentences on what it does and why it's interesting[/DESCRIPTION]
[WHY]1-2 sentences on why this project specifically forces deep learning of the technology[/WHY]
[FEATURES]feature 1 | feature 2 | feature 3[/FEATURES]
[LEARN]specific skill 1 | specific skill 2 | specific skill 3[/LEARN]
[TOOLS]the requested technology | other tools needed[/TOOLS]
[FIRST_STEP]The very first concrete thing to build — specific enough to start today[/FIRST_STEP]
[TIME]Realistic time range to complete[/TIME]
    """

    return StreamingResponse(stream_anthropic(prompt, 1000), media_type="text/event-stream")

@app.get("/discover")
def discover_ideas():
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    # 1. Scrape GitHub Trending
    try:
        github_res = requests.get("https://github.com/trending")
        github_res.raise_for_status()
        soup = BeautifulSoup(github_res.text, "html.parser")
        repos_html = soup.select("article.Box-row")[:10]
        trending_repos = []
        for repo in repos_html:
            name_elem = repo.select_one("h2 a")
            name = name_elem.text.strip().replace(" ", "").replace("\n", "") if name_elem else "Unknown"
            desc_elem = repo.select_one("p")
            description = desc_elem.text.strip() if desc_elem else "No description"
            lang_elem = repo.select_one("[itemprop='programmingLanguage']")
            language = lang_elem.text.strip() if lang_elem else "Unknown"
            trending_repos.append({"name": name, "description": description, "language": language})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to scrape GitHub trending: {str(e)}")

    # 2. Hacker News Top Stories
    try:
        hn_url = "https://hacker-news.firebaseio.com/v0/topstories.json"
        hn_res = requests.get(hn_url)
        hn_res.raise_for_status()
        top_ids = hn_res.json()[:10]
        
        tech_keywords = ['tech', 'software', 'dev', 'ai', 'code', 'data', 'web', 'app', 'api', 'cloud', 'linux', 'startup', 'security', 'programming', 'release', 'framework', 'vulnerability', 'server', 'database', 'python', 'javascript', 'rust', 'github', 'open source', 'llm', 'model', 'cybersecurity', 'engineering', 'machine learning', 'algorithm']
        
        hn_stories = []
        for story_id in top_ids:
            story_res = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json")
            story_data = story_res.json()
            if story_data and 'title' in story_data:
                title = story_data['title']
                if any(kw in title.lower() for kw in tech_keywords):
                    hn_stories.append(title)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch Hacker News: {str(e)}")

    # 3. Format Context
    context_string = "Trending GitHub Repos:\n"
    for repo in trending_repos:
        context_string += f"- {repo['name']} ({repo['language']}): {repo['description']}\n"
    context_string += "\nTop Hacker News Tech Stories:\n"
    for story in hn_stories:
        context_string += f"- {story}\n"

    # 4. Generate Ideas from Context
    prompt = f"""You are an expert software engineer and creative project mentor who stays 
on the cutting edge of technology.

You will be given a list of currently trending GitHub repositories and 
top Hacker News stories. Your job is to synthesize what's genuinely 
hot right now and generate 3 creative, impressive project ideas that 
a developer could build to immerse themselves in these emerging tools 
and technologies.

Each idea must:
- Be directly inspired by or built with something from the trending data provided
- Be creative and non-obvious — not just "build a wrapper around X"
- Be scoped for one developer to complete in a weekend to two weeks
- Force deep hands-on learning with a technology that is actually trending right now
- Be genuinely useful or interesting, not just a demo

Here is what is currently trending:

{context_string}

Return exactly 3 project ideas. Format each idea using ONLY the following exact format, field by field, using these simple delimiters:

[TITLE]Short punchy project name[/TITLE]
[TAGLINE]One sentence that makes someone want to build it[/TAGLINE]
[INSPIRED_BY]Which trending repo or story inspired this[/INSPIRED_BY]
[DESCRIPTION]2-3 sentences on what it does and why it's interesting[/DESCRIPTION]
[WHY]Why this forces real learning of something trending[/WHY]
[FEATURES]feature 1 | feature 2 | feature 3[/FEATURES]
[TOOLS]primary trending tech | other tools[/TOOLS]
[LEARN]specific skill 1 | specific skill 2 | specific skill 3[/LEARN]
[FIRST_STEP]The very first concrete thing to build today[/FIRST_STEP]
[TIME]Realistic time range[/TIME]

Do this 3 times consecutively, once for each idea. Do not include any other text outside these tags.
    """

    return StreamingResponse(stream_anthropic(prompt, 2000), media_type="text/event-stream")

@app.post("/save-idea")
def save_idea(idea: SavedIdea):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    
    data = idea.dict()
    
    # Supabase/PostgreSQL is expecting an Array for these columns since you made them array fields.
    if isinstance(data.get("what_youll_learn"), str):
        data["what_youll_learn"] = [x.strip() for x in data["what_youll_learn"].split("|") if x.strip()]
    if isinstance(data.get("tools_and_tech"), str):
        data["tools_and_tech"] = [x.strip() for x in data["tools_and_tech"].split("|") if x.strip()]

    try:
        res = supabase.table("saved_ideas").insert(data).execute()
        return res.data[0] if res.data else {"status": "error"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/saved-ideas")
def get_saved_ideas():
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    res = supabase.table("saved_ideas").select("*").order("created_at", desc=True).execute()
    return res.data

@app.delete("/saved-ideas/{id}")
def delete_saved_idea(id: str):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    supabase.table("saved_ideas").delete().eq("id", id).execute()
    return {"status": "success"}

@app.get("/")
def read_root():
    # Read index.html relative to this file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(base_dir, "index.html")
    with open(index_path, "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

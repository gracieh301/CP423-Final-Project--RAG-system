"""
SpongeBob Wiki Crawler — Episodes + Characters (Seasons 1-10)

Crawls episode pages by season, and for each episode, discovers and crawls
the character pages linked from its infobox — deduplicating characters
across episodes so each one is only fetched/stored once.

Output: spongebob_corpus.json
"""

import json
import time
import re
from urllib.parse import unquote

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://spongebob.fandom.com"
API_URL = f"{BASE_URL}/api.php"

HEADERS = {
    "User-Agent": "RAG-Class-Assignment-Bot/1.0 (contact: your_email@school.edu)"
}

REQUEST_DELAY = 0.3




def get_episode_pages(season):
    """Return a list of episode page titles for a given season, via the
    MediaWiki category system (e.g. Category:Season 1 episodes)."""

    category = f"Category:Season {season} episodes"
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmlimit": "max",
        "cmnamespace": 0,   # main namespace only, excludes File:/Category: etc.
        "format": "json",
    }

    titles = []
    while True:
        r = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()

        members = data.get("query", {}).get("categorymembers", [])
        titles.extend(m["title"] for m in members)

        if "continue" in data:
            params.update(data["continue"])
        else:
            break

    return titles


def get_page_html(title):
    """Fetch rendered HTML for a page title via the API's parse action.
    Returns None if the page doesn't exist or the request fails."""

    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "redirects": 1,
        "format": "json",
    }

    try:
        r = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.RequestException:
        return None

    if "error" in data:
        return None

    return data.get("parse", {}).get("text", {}).get("*")


EXCLUDED_NAMESPACES = (
    "Special", "File", "Category", "Template", "User", "Help",
)

# Non-character "characters" that commonly show up in the list but
# aren't individual character pages we want to crawl as documents.
EXCLUDED_TITLES = (
    "List of incidental characters",
)


def get_characters_from_episode(html):
    """Parse an episode page's HTML and pull character name/page pairs
    from the 'Characters' section list (a <ul> following the
    <h3 id="Characters"> headline), not the infobox."""

    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")

    # Find the headline span with id="Characters" (inside an h2/h3),
    # then walk forward to the next <ul> sibling, which holds the list.
    headline = soup.find(id="Characters")
    if not headline:
        return []

    heading_tag = headline.find_parent(["h2", "h3", "h4"])
    if not heading_tag:
        return []

    char_list = heading_tag.find_next_sibling("ul")
    if not char_list:
        return []

    characters = []
    seen_pages = set()

    # find_all(a) here also picks up links nested inside the "Incidentals"
    # sub-<ul> (e.g. Incidental 30, Incidental 30A) which is intentional 
    # they get treated as their own character pages.
    for link in char_list.find_all("a", href=True):
        href = link["href"]
        if "/wiki/" not in href:
            continue

        page_title = unquote(href.split("/wiki/", 1)[-1]).replace("_", " ")

        first_segment = page_title.split(":")[0]
        if first_segment in EXCLUDED_NAMESPACES:
            continue
        if page_title in EXCLUDED_TITLES:
            continue

        if page_title in seen_pages:
            continue
        seen_pages.add(page_title)

        display_name = link.get("title", page_title) or page_title
        characters.append({"page": page_title, "title": display_name})

    return characters



def _html_to_text(html):
    """Convert page HTML into readable plaintext, stripping nav/edit cruft."""

    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")

    # remove elements that aren't real article content
    for tag in soup.find_all(["table", "sup", "style", "script"]):
        tag.decompose()
    for tag in soup.find_all(class_=re.compile("editsection|reference|toc")):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def create_character_document(character_id, title, html):
    return {
        "id": character_id,
        "type": "character",
        "title": title,
        "url": f"{BASE_URL}/wiki/{title.replace(' ', '_')}",
        "text": _html_to_text(html),
    }


def create_episode_document(episode_id, title, html, characters):
    return {
        "id": episode_id,
        "type": "episode",
        "title": title,
        "url": f"{BASE_URL}/wiki/{title.replace(' ', '_')}",
        "characters": characters,
        "text": _html_to_text(html),
    }



# Main


def crawl_spongebob():

    documents = []
    crawled_episodes = set()
    crawled_characters = {}
    character_counter = 1

    for season in range(1, 11):

        print("\n" + "=" * 60)
        print(f"SEASON {season}")
        print("=" * 60)

        episode_pages = get_episode_pages(season)
        print(f"Found {len(episode_pages)} possible episode pages.")

        episode_number = 1

        for episode_title in episode_pages:

            if episode_title in crawled_episodes:
                continue

            print(f"\n[{season:02d} / {episode_number:02d}] {episode_title}")

            episode_html = get_page_html(episode_title)
            if not episode_html:
                print("    Failed to retrieve episode.")
                continue

            characters = get_characters_from_episode(episode_html)
            episode_characters = []
            print(f"    Characters listed: {len(characters)}")

            for character in characters:

                character_page = character["page"]
                character_title = character["title"]

                if character_page in crawled_characters:
                    character_id = crawled_characters[character_page]
                    print(f"      SKIP: {character_title} ({character_id})")

                else:
                    print(f"      Crawling character: {character_title}")

                    character_html = get_page_html(character_page)
                    if not character_html:
                        print(f"      Failed: {character_title}")
                        continue

                    character_id = f"CHAR{character_counter}"

                    character_document = create_character_document(
                        character_id, character_title, character_html
                    )
                    documents.append(character_document)

                    crawled_characters[character_page] = character_id
                    character_counter += 1

                    print(f"      Added: {character_id}")
                    time.sleep(REQUEST_DELAY)

                episode_characters.append({
                    "id": character_id,
                    "name": character_title,
                })

            episode_id = f"S{season:02d}E{episode_number:02d}"

            episode_document = create_episode_document(
                episode_id, episode_title, episode_html, episode_characters
            )
            documents.append(episode_document)

            crawled_episodes.add(episode_title)
            print(f"    Added episode: {episode_id}")

            episode_number += 1
            time.sleep(REQUEST_DELAY)

    corpus = {
        "corpus_name": "SpongeBob SquarePants Seasons 1-10",
        "documents": documents,
    }

    with open("spongebob_corpus.json", "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    episode_count = sum(doc["type"] == "episode" for doc in documents)
    character_count = sum(doc["type"] == "character" for doc in documents)

    print("\n" + "=" * 60)
    print("CRAWLING COMPLETE")
    print("=" * 60)
    print(f"Episodes:   {episode_count}")
    print(f"Characters: {character_count}")
    print(f"Total:      {len(documents)}")
    print("\nOutput: spongebob_corpus.json")


if __name__ == "__main__":
    crawl_spongebob()
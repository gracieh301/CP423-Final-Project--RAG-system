import requests
import json
import time
import re


API_URL = "https://spongebob.fandom.com/api.php"

HEADERS = {
    "User-Agent": "SpongeBob-RAG-Project/1.0"
}


# ==========================================================
# 1. GET ALL PAGES FROM A CATEGORY
# ==========================================================

def get_category_pages(category):

    pages = []

    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": "max",
        "cmnamespace": 0
    }

    while True:

        response = requests.get(
            API_URL,
            params=params,
            headers=HEADERS,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        for page in data["query"]["categorymembers"]:
            pages.append(page["title"])

        # MediaWiki pagination
        if "continue" not in data:
            break

        params.update(data["continue"])

    return pages


# ==========================================================
# 2. GET PAGE CONTENT
# ==========================================================

def get_page_texts(titles):

    params = {
        "action": "query",
        "format": "json",
        "titles": "|".join(titles),
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main"
    }

    response = requests.get(
        API_URL,
        params=params,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    documents = []

    for page in data["query"]["pages"].values():

        if "missing" in page:
            continue

        revisions = page.get("revisions", [])

        if not revisions:
            continue

        content = (
            revisions[0]
            .get("slots", {})
            .get("main", {})
            .get("*", "")
        )

        documents.append({
            "title": page["title"],
            "text": content
        })

    return documents


# ==========================================================
# 3. REMOVE NESTED MEDIAWIKI TEMPLATES
# ==========================================================

def remove_templates(text):

    while "{{" in text:

        start = text.find("{{")

        depth = 0
        end = None

        for i in range(start, len(text) - 1):

            if text[i:i + 2] == "{{":
                depth += 1

            elif text[i:i + 2] == "}}":

                depth -= 1

                if depth == 0:
                    end = i + 2
                    break

        if end is None:
            break

        text = text[:start] + text[end:]

    return text


# ==========================================================
# 4. CLEAN WIKITEXT
# ==========================================================

def clean_wikitext(text):

    # ------------------------------------------------------
    # Remove sections that aren't useful for RAG
    # ------------------------------------------------------
    text = re.sub(
    r"^Characters\s*$[\s\S]*?(?=^Synopsis\s*$)",
    "",
    text,
    flags=re.MULTILINE | re.IGNORECASE
)

    # Remove References section and everything after it
    text = re.split(
        r"^==\s*References\s*==",
        text,
        flags=re.MULTILINE | re.IGNORECASE
    )[0]

    # Remove Videos sections
    text = re.split(
        r"^==\s*Videos\s*==",
        text,
        flags=re.MULTILINE | re.IGNORECASE
    )[0]

    # Remove Names in other languages
    text = re.split(
        r"^==\s*Names in other languages\s*==",
        text,
        flags=re.MULTILINE | re.IGNORECASE
    )[0]


    # ------------------------------------------------------
    # Remove galleries
    # ------------------------------------------------------

    text = re.sub(
        r"<gallery.*?>.*?</gallery>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )


    # ------------------------------------------------------
    # Remove HTML tags
    # ------------------------------------------------------

    text = re.sub(
        r"<[^>]+>",
        "",
        text
    )


    # ------------------------------------------------------
    # Remove image/file markup
    # ------------------------------------------------------

    text = re.sub(
        r"\[\[File:.*?\]\]",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\[\[Image:.*?\]\]",
        "",
        text,
        flags=re.IGNORECASE
    )


    # ------------------------------------------------------
    # Remove external URLs
    # ------------------------------------------------------

    text = re.sub(
        r"https?://\S+",
        "",
        text
    )


    # ------------------------------------------------------
    # Remove templates
    # ------------------------------------------------------

    text = remove_templates(text)





    # ------------------------------------------------------
    # Convert MediaWiki links
    #
    # [[SpongeBob SquarePants]]
    # -> SpongeBob SquarePants
    #
    # [[SpongeBob SquarePants|SpongeBob]]
    # -> SpongeBob
    # ------------------------------------------------------

    def replace_link(match):

        content = match.group(1)

        # Ignore files/images
        if content.lower().startswith(
            ("file:", "image:", "media:")
        ):
            return ""

        # Handle links with display text
        if "|" in content:
            return content.split("|")[-1]

        # Handle section links
        if "#" in content:
            return content.split("#")[0]

        return content


    text = re.sub(
        r"\[\[(.*?)\]\]",
        replace_link,
        text
    )


    # ------------------------------------------------------
    # Remove external link syntax
    #
    # [https://example.com Example]
    # -> Example
    # ------------------------------------------------------

    text = re.sub(
        r"\[https?://[^\s\]]+\s*([^\]]*)\]",
        r"\1",
        text
    )


    # ------------------------------------------------------
    # Remove bold / italic markup
    # ------------------------------------------------------

    text = text.replace("'''", "")
    text = text.replace("''", "")


    # ------------------------------------------------------
    # Remove headings markup but keep heading text
    #
    # ===Synopsis===
    # -> Synopsis
    # ------------------------------------------------------

    text = re.sub(
        r"^={2,6}\s*(.*?)\s*={2,6}\s*$",
        r"\1",
        text,
        flags=re.MULTILINE
    )


    # ------------------------------------------------------
    # Remove MediaWiki directives
    # ------------------------------------------------------

    text = re.sub(
        r"^#.*$",
        "",
        text,
        flags=re.MULTILINE
    )


    # ------------------------------------------------------
    # Clean bullet points
    # ------------------------------------------------------

    text = re.sub(
        r"^[*#:]+\s*",
        "- ",
        text,
        flags=re.MULTILINE
    )


    # ------------------------------------------------------
    # Remove excessive whitespace
    # ------------------------------------------------------

    text = re.sub(
        r"\n\s*\n\s*\n+",
        "\n\n",
        text
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )


    # ------------------------------------------------------
    # Remove empty lines around whitespace
    # ------------------------------------------------------

    text = "\n".join(
        line.strip()
        for line in text.splitlines()
        if line.strip()
    )


    return text.strip()

# ==========================================================
# 5. EXTRACT CHARACTERS
# ==========================================================

def extract_characters(text):

    # Find the Characters section
    match = re.search(
        r"^Characters\s*$([\s\S]*?)(?=^Synopsis\s*$)",
        text,
        flags=re.MULTILINE | re.IGNORECASE
    )

    if not match:
        return []

    characters_section = match.group(1)

    characters = []

    for line in characters_section.splitlines():

        line = line.strip()

        # Character entries are bullet points
        if line.startswith("-"):
            
            # Remove one or more leading "- "
            character = re.sub(
                r"^-+\s*",
                "",
                line
            ).strip()

            if character:
                characters.append(character)

    return characters


# ==========================================================
# 5. FIND EPISODES FROM SEASONS 1-10
# ==========================================================

episode_pages = {}

print("Finding episodes...")
print()

for season in range(1, 11):

    category = f"Season {season} episodes"

    pages = get_category_pages(category)

    print(
        f"Season {season}: "
        f"{len(pages)} category members"
    )

    for page in pages:

        # Keep track of the season
        if page not in episode_pages:
            episode_pages[page] = season


print()
print(
    f"Unique episode category members: "
    f"{len(episode_pages)}"
)


# ==========================================================
# 6. RETRIEVE EPISODE TEXT
# ==========================================================

documents = []

titles = list(episode_pages.keys())

BATCH_SIZE = 20

print()
print("Downloading page content...")
print()

for i in range(0, len(titles), BATCH_SIZE):

    batch = titles[i:i + BATCH_SIZE]

    print(
        f"Downloading "
        f"{i + 1}-{i + len(batch)} "
        f"of {len(titles)}..."
    )

    try:

        batch_documents = get_page_texts(batch)

        for document in batch_documents:

            raw_text = document["text"]

            # Extract characters before cleaning the text
            characters = extract_characters(raw_text)

            # Clean the raw wikitext
            cleaned_text = clean_wikitext(raw_text)

            # Skip empty pages
            if not cleaned_text:
                continue

            document["characters"] = characters
            document["text"] = cleaned_text
            document["type"] = "episode"
            document["season"] = episode_pages[
                document["title"]
            ]

            documents.append(document)

    except requests.RequestException as e:

        print(f"Request failed: {e}")

    time.sleep(0.5)


# ==========================================================
# 7. REMOVE DUPLICATES
# ==========================================================

unique_documents = {}

for document in documents:

    unique_documents[document["title"]] = document


documents = list(unique_documents.values())


# ==========================================================
# 8. SAVE RAW/CLEAN DATA
# ==========================================================

with open(
    "spongebob_documents_clean.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        documents,
        f,
        ensure_ascii=False,
        indent=2
    )


# ==========================================================
# 9. STATISTICS
# ==========================================================

print()
print("=" * 60)
print("SCRAPING COMPLETE")
print("=" * 60)

print(
    f"Total episode documents: "
    f"{len(documents)}"
)

print()

for season in range(1, 11):

    count = sum(
        1
        for doc in documents
        if doc["season"] == season
    )

    print(
        f"Season {season}: {count} documents"
    )

print()
print(
    "Saved to: "
    "spongebob_documents_clean.json"
)
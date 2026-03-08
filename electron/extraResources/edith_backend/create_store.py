import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise SystemExit("GOOGLE_API_KEY is missing. Set it in .env.")

client = genai.Client(api_key=api_key)

display = os.environ.get("EDITH_STORE_DISPLAY", "edith-main")

existing = {s.display_name: s.name for s in client.file_search_stores.list()}
if display in existing:
    store_name = existing[display]
    print(f"FOUND {display}: {store_name}")
else:
    s = client.file_search_stores.create(config={"display_name": display})
    store_name = s.name
    print(f"CREATED {display}: {store_name}")

print("\nAdd this to .env:")
print(f"EDITH_STORE_ID={store_name}")

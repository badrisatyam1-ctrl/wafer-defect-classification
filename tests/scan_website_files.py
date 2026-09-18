import re
import os
from pathlib import Path

def scan_website():
    print("=" * 60)
    print("SCANNING WEBSITE INTEGRITY: HTML, CSS, JS, API")
    print("=" * 60)

    static_dir = Path("deployment/static")
    html_path = static_dir / "index.html"
    js_path = static_dir / "script.js"
    css_path = static_dir / "styles.css"

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    with open(js_path, "r", encoding="utf-8") as f:
        js_content = f.read()

    with open(css_path, "r", encoding="utf-8") as f:
        css_content = f.read()

    # 1. Check ID references
    ids_in_js = set(re.findall(r"getElementById\(['\"]([^'\"]+)['\"]", js_content))
    ids_in_html = set(re.findall(r"id=['\"]([^'\"]+)['\"]", html_content))

    print(f"Total IDs in JS: {len(ids_in_js)}")
    print(f"Total IDs in HTML: {len(ids_in_html)}")

    missing_ids = ids_in_js - ids_in_html
    if missing_ids:
        print("\n[!] MISSING IDs IN HTML:")
        for m in sorted(missing_ids):
            print(f"  - {m}")
    else:
        print("[OK] All JS element IDs matched in HTML.")

    # 2. Check onclick handlers in HTML vs functions in JS
    onclicks = set(re.findall(r"onclick=['\"]([a-zA-Z0-9_]+)\(", html_content))
    print(f"\nTotal onclick handlers in HTML: {len(onclicks)}")
    missing_funcs = []
    for fn in onclicks:
        # Check if function fn exists in JS
        pattern = rf"(function\s+{fn}\s*\(|const\s+{fn}\s*=|let\s+{fn}\s*=|var\s+{fn}\s*=|window\.{fn}\s*=)"
        if not re.search(pattern, js_content):
            missing_funcs.append(fn)

    if missing_funcs:
        print("[!] MISSING FUNCTIONS IN JS:")
        for fn in missing_funcs:
            print(f"  - {fn}")
    else:
        print("[OK] All onclick handler functions exist in JS.")

    # 3. Check for any broken links or tags in HTML
    # Check CSS classes used in HTML that might be missing in CSS
    classes_in_html = set()
    for m in re.findall(r"class=['\"]([^'\"]+)['\"]", html_content):
        for c in m.split():
            classes_in_html.add(c)
    print(f"\nTotal CSS classes used in HTML: {len(classes_in_html)}")

    # 4. Check API endpoints called in JS vs defined in server.py
    with open("deployment/server.py", "r", encoding="utf-8") as f:
        server_content = f.read()

    js_endpoints = set(re.findall(r"fetch\(['\"]([^'\"]+)['\"]", js_content))
    server_endpoints = set(re.findall(r"@app\.(?:get|post|put|delete)\(['\"]([^'\"]+)['\"]", server_content))

    print(f"\nJS fetch endpoints: {sorted(list(js_endpoints))}")
    print(f"Server defined endpoints: {sorted(list(server_endpoints))}")

    unmatched = [ep for ep in js_endpoints if not ep.startswith("data:") and ep not in server_endpoints]
    if unmatched:
        print("[!] Unmatched endpoints called by JS:")
        for ep in unmatched:
            print(f"  - {ep}")
    else:
        print("[OK] All JS fetch endpoints match server API routes.")

    print("\n" + "=" * 60)
    print("SCAN COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    scan_website()

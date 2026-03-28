"""🔦 Apple Spotlight search via mdfind CLI + metadata."""

from typing import Dict, Any
from strands import tool


@tool
def apple_spotlight(
    action: str = "search",
    query: str = "",
    folder: str = None,
    kind: str = None,
    limit: int = 20,
    path: str = None,
) -> Dict[str, Any]:
    """🔦 Spotlight search — find files, apps, documents instantly.

    Args:
        action: Action to perform:
            - "search": Full-text search (like Cmd+Space)
            - "name": Search by filename
            - "kind": Search by file kind (pdf, image, music, etc.)
            - "metadata": Get metadata for a specific file
            - "recent": Recently modified files
        query: Search query
        folder: Limit search to this folder
        kind: File kind filter (pdf, image, movie, music, app, folder, email, presentation, spreadsheet)
        limit: Max results (default: 20)
        path: File path for metadata action

    Returns:
        Dict with search results
    """
    import subprocess

    if action == "search":
        return _search(query, folder, kind, limit)
    elif action == "name":
        return _search_name(query, folder, limit)
    elif action == "kind":
        return _search_kind(kind or query, folder, limit)
    elif action == "metadata":
        if not path:
            return {"status": "error", "content": [{"text": "path required for metadata"}]}
        return _get_metadata(path)
    elif action == "recent":
        return _recent_files(folder, kind, limit)
    else:
        return {"status": "error", "content": [{"text": f"Unknown action: {action}. Use: search, name, kind, metadata, recent"}]}


def _search(query, folder, kind, limit):
    """Full-text Spotlight search."""
    try:
        import subprocess

        if not query:
            return {"status": "error", "content": [{"text": "query required for search"}]}

        cmd = ["mdfind"]
        if folder:
            cmd.extend(["-onlyin", folder])

        # Build query string
        mdfind_query = query
        if kind:
            kind_map = {
                "pdf": "kMDItemContentType == 'com.adobe.pdf'",
                "image": "kMDItemContentTypeTree == 'public.image'",
                "movie": "kMDItemContentTypeTree == 'public.movie'",
                "music": "kMDItemContentTypeTree == 'public.audio'",
                "app": "kMDItemContentType == 'com.apple.application-bundle'",
                "folder": "kMDItemContentType == 'public.folder'",
                "email": "kMDItemContentTypeTree == 'public.message'",
                "presentation": "kMDItemContentTypeTree == 'public.presentation'",
                "spreadsheet": "kMDItemContentTypeTree == 'public.spreadsheet'",
            }
            if kind in kind_map:
                mdfind_query = f"{kind_map[kind]} && kMDItemTextContent == '*{query}*'"

        cmd.append(mdfind_query)

        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        results = [line for line in r.stdout.strip().split("\n") if line][:limit]

        if not results:
            return {"status": "success", "content": [{"text": f"🔦 No results for '{query}'."}]}

        lines = [f"🔦 Spotlight: {len(results)} results for '{query}':\n"]
        for path in results:
            # Get file size
            import os
            try:
                size = os.path.getsize(path)
                if size > 1_000_000:
                    size_str = f"{size / 1_000_000:.1f}MB"
                elif size > 1_000:
                    size_str = f"{size / 1_000:.0f}KB"
                else:
                    size_str = f"{size}B"
            except:
                size_str = "?"

            name = os.path.basename(path)
            parent = os.path.dirname(path)
            lines.append(f"  📄 {name} ({size_str})")
            lines.append(f"     {parent}")

        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    except Exception as e:
        return {"status": "error", "content": [{"text": f"Search error: {e}"}]}


def _search_name(query, folder, limit):
    """Search by filename."""
    try:
        import subprocess

        cmd = ["mdfind"]
        if folder:
            cmd.extend(["-onlyin", folder])
        cmd.extend(["-name", query])

        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        results = [line for line in r.stdout.strip().split("\n") if line][:limit]

        if not results:
            return {"status": "success", "content": [{"text": f"🔦 No files named '{query}'."}]}

        import os
        lines = [f"🔦 Files matching name '{query}' ({len(results)}):\n"]
        for path in results:
            name = os.path.basename(path)
            parent = os.path.dirname(path)
            lines.append(f"  📄 {name}")
            lines.append(f"     {parent}")

        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    except Exception as e:
        return {"status": "error", "content": [{"text": f"Name search error: {e}"}]}


def _search_kind(kind, folder, limit):
    """Search by file kind."""
    try:
        import subprocess

        kind_map = {
            "pdf": "kMDItemContentType == 'com.adobe.pdf'",
            "image": "kMDItemContentTypeTree == 'public.image'",
            "movie": "kMDItemContentTypeTree == 'public.movie'",
            "music": "kMDItemContentTypeTree == 'public.audio'",
            "app": "kMDItemContentType == 'com.apple.application-bundle'",
            "folder": "kMDItemContentType == 'public.folder'",
            "email": "kMDItemContentTypeTree == 'public.message'",
            "presentation": "kMDItemContentTypeTree == 'public.presentation'",
            "spreadsheet": "kMDItemContentTypeTree == 'public.spreadsheet'",
            "document": "kMDItemContentTypeTree == 'public.composite-content'",
            "source": "kMDItemContentTypeTree == 'public.source-code'",
        }

        mdfind_query = kind_map.get(kind, f"kMDItemKind == '*{kind}*'")

        cmd = ["mdfind"]
        if folder:
            cmd.extend(["-onlyin", folder])
        cmd.append(mdfind_query)

        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        results = [line for line in r.stdout.strip().split("\n") if line][:limit]

        if not results:
            return {"status": "success", "content": [{"text": f"🔦 No '{kind}' files found."}]}

        import os
        lines = [f"🔦 '{kind}' files ({len(results)}):\n"]
        for path in results:
            name = os.path.basename(path)
            lines.append(f"  📄 {name}")
            lines.append(f"     {path}")

        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    except Exception as e:
        return {"status": "error", "content": [{"text": f"Kind search error: {e}"}]}


def _get_metadata(path):
    """Get Spotlight metadata for a file."""
    try:
        import subprocess, os

        path = os.path.expanduser(path)
        if not os.path.exists(path):
            return {"status": "error", "content": [{"text": f"File not found: {path}"}]}

        r = subprocess.run(
            ["mdls", path],
            capture_output=True, text=True, timeout=10
        )

        lines = [f"🔦 Metadata for {os.path.basename(path)}:\n"]
        # Parse interesting fields
        for line in r.stdout.strip().split("\n"):
            line = line.strip()
            if "= (null)" in line or not line:
                continue
            if any(k in line for k in ["kMDItemDisplayName", "kMDItemContentType",
                "kMDItemFSSize", "kMDItemContentCreationDate", "kMDItemContentModificationDate",
                "kMDItemKind", "kMDItemPixelHeight", "kMDItemPixelWidth",
                "kMDItemDurationSeconds", "kMDItemAuthors", "kMDItemTitle",
                "kMDItemWhereFroms", "kMDItemCodecs"]):
                lines.append(f"  {line}")

        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    except Exception as e:
        return {"status": "error", "content": [{"text": f"Metadata error: {e}"}]}


def _recent_files(folder, kind, limit):
    """Find recently modified files."""
    try:
        import subprocess

        query = "kMDItemFSContentChangeDate >= $time.today(-7)"
        if kind:
            kind_map = {
                "pdf": "kMDItemContentType == 'com.adobe.pdf'",
                "image": "kMDItemContentTypeTree == 'public.image'",
                "source": "kMDItemContentTypeTree == 'public.source-code'",
            }
            if kind in kind_map:
                query += f" && {kind_map[kind]}"

        cmd = ["mdfind"]
        if folder:
            cmd.extend(["-onlyin", folder])
        cmd.append(query)

        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        results = [line for line in r.stdout.strip().split("\n") if line][:limit]

        if not results:
            return {"status": "success", "content": [{"text": "🔦 No recent files found."}]}

        import os
        from datetime import datetime

        lines = [f"🔦 Recently modified files ({len(results)}):\n"]
        for path in results:
            try:
                name = os.path.basename(path)
                mtime = os.path.getmtime(path)
                dt = datetime.fromtimestamp(mtime)
                lines.append(f"  📄 {name} — {dt.strftime('%b %d %H:%M')}")
                lines.append(f"     {path}")
            except:
                lines.append(f"  📄 {path}")

        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    except Exception as e:
        return {"status": "error", "content": [{"text": f"Recent files error: {e}"}]}

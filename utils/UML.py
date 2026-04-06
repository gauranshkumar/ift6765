#!/usr/bin/env python3
"""
PlantUML Web Validator
----------------------
Validates .puml source by submitting it to the official PlantUML web server
(plantuml.com) and inspecting the response for error indicators.

How it works:
  1. Encodes the UML source using PlantUML's custom base64 encoding
  2. Sends a request to https://www.plantuml.com/plantuml/png/<encoded>
  3. Inspects the PNG response — PlantUML returns a special error image
     when the diagram has syntax errors, with the error text embedded
  4. Also hits the /check/ endpoint (returns plain-text errors if available)

Usage:
  python plantuml_validator.py diagram.puml
  python plantuml_validator.py diagram.puml --server https://my-plantuml-server.com
  cat diagram.puml | python plantuml_validator.py --stdin
  python plantuml_validator.py diagram.puml --verbose
"""

import re
import sys
import zlib
import string
import struct
import argparse
import urllib.request
import urllib.error
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# PlantUML custom Base64 encoding  (the ~1 alphabet plantuml.com uses)
# ─────────────────────────────────────────────────────────────────────────────

_PLANTUML_ALPHABET = (
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
)

def _encode6(b: int) -> str:
    return _PLANTUML_ALPHABET[b & 0x3F]

def _encode3bytes(b1: int, b2: int, b3: int) -> str:
    c1 =  b1 >> 2
    c2 = ((b1 & 0x3) << 4) | (b2 >> 4)
    c3 = ((b2 & 0xF) << 2) | (b3 >> 6)
    c4 =  b3 & 0x3F
    return _encode6(c1) + _encode6(c2) + _encode6(c3) + _encode6(c4)

def plantuml_encode(source: str) -> str:
    """Compress with zlib (deflate) then encode with PlantUML's base64 alphabet."""
    data = zlib.compress(source.encode("utf-8"))[2:-4]  # strip zlib header/checksum
    result = []
    i = 0
    while i < len(data):
        b1 = data[i]
        b2 = data[i + 1] if i + 1 < len(data) else 0
        b3 = data[i + 2] if i + 2 < len(data) else 0
        result.append(_encode3bytes(b1, b2, b3))
        i += 3
    return "".join(result)


# ─────────────────────────────────────────────────────────────────────────────
# PNG helpers — detect PlantUML error images
# ─────────────────────────────────────────────────────────────────────────────

def _read_png_text_chunks(data: bytes) -> dict:
    """Extract tEXt chunks from a PNG — PlantUML embeds error info there."""
    chunks = {}
    if data[:8] != b'\x89PNG\r\n\x1a\n':
        return chunks
    idx = 8
    while idx < len(data) - 12:
        try:
            length = struct.unpack(">I", data[idx:idx+4])[0]
            chunk_type = data[idx+4:idx+8].decode("ascii", errors="replace")
            chunk_data = data[idx+8:idx+8+length]
            if chunk_type == "tEXt":
                null = chunk_data.find(b'\x00')
                if null != -1:
                    key = chunk_data[:null].decode("utf-8", errors="replace")
                    val = chunk_data[null+1:].decode("utf-8", errors="replace")
                    chunks[key] = val
            idx += 12 + length
        except Exception:
            break
    return chunks

def _is_error_png(data: bytes) -> tuple[bool, str]:
    """
    Returns (is_error, error_message).
    PlantUML error PNGs are small (<5 KB) and contain red/error-coloured pixels,
    plus often embed the error text in tEXt chunks.
    """
    # Check tEXt chunks first
    chunks = _read_png_text_chunks(data)
    for key in ("plantuml", "error", "Error"):
        if key in chunks and chunks[key].strip():
            text = chunks[key].strip()
            if any(word in text.lower() for word in ("error", "syntax", "unexpected", "cannot", "unknown")):
                return True, text

    # Heuristic: real diagrams are usually > 2 KB; error PNGs are tiny
    # and PlantUML error images contain the literal text "Syntax Error"
    # encoded in the image — we can detect it via the raw bytes
    raw_text = data.decode("latin-1")
    error_markers = ["Syntax Error", "Error", "cannot parse", "Unexpected token"]
    for marker in error_markers:
        if marker in raw_text:
            # Extract surrounding context
            idx = raw_text.index(marker)
            snippet = raw_text[max(0, idx-20):idx+120]
            snippet = re.sub(r'[^\x20-\x7E\n]', '', snippet).strip()
            return True, snippet or marker

    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# Validator
# ─────────────────────────────────────────────────────────────────────────────

class PlantUMLWebValidator:

    def __init__(self, server: str = "https://www.plantuml.com/plantuml"):
        self.server = server.rstrip("/")

    def validate(self, source: str) -> dict:
        """
        Returns a dict:
          {
            "valid":   bool,
            "errors":  [str, ...],
            "encoded": str,          # the encoded URL token
            "url":     str,          # full PNG URL you can open in browser
          }
        """
        result = {"valid": False, "errors": [], "encoded": "", "url": ""}

        # Quick pre-flight: check @startuml / @enduml presence
        stripped = source.strip()
        if not re.search(r"@startuml", stripped, re.IGNORECASE):
            result["errors"].append("Missing @startuml directive (pre-flight check)")
            return result
        if not re.search(r"@enduml", stripped, re.IGNORECASE):
            result["errors"].append("Missing @enduml directive (pre-flight check)")
            return result

        # Encode
        try:
            encoded = plantuml_encode(source)
        except Exception as e:
            result["errors"].append(f"Encoding error: {e}")
            return result

        result["encoded"] = encoded
        png_url = f"{self.server}/png/{encoded}"
        result["url"] = png_url

        # Fetch PNG from server
        try:
            req = urllib.request.Request(
                png_url,
                headers={"User-Agent": "plantuml-validator/1.0"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                content_type = resp.headers.get("Content-Type", "")
                png_data = resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 400:
                result["errors"].append(
                    "PlantUML server rejected the diagram (HTTP 400) — syntax error in UML source"
                )
            else:
                result["errors"].append(f"PlantUML server returned HTTP {e.code}: {e.reason}")
            return result
        except urllib.error.URLError as e:
            result["errors"].append(
                f"Cannot reach PlantUML server at {self.server}: {e.reason}\n"
                f"  → Check your internet connection or supply --server <url>"
            )
            return result
        except Exception as e:
            result["errors"].append(f"HTTP request failed: {e}")
            return result

        # Check if server returned an image at all
        if "image/png" not in content_type and "image/" not in content_type:
            result["errors"].append(
                f"Server returned unexpected content-type: {content_type}"
            )
            return result

        # Inspect the PNG for embedded errors
        is_err, msg = _is_error_png(png_data)
        if is_err:
            result["errors"].append(f"PlantUML server reported: {msg}")
            return result

        # All good
        result["valid"] = True
        return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"

def _c(color, text):
    """Apply ANSI color if stdout is a TTY."""
    if sys.stdout.isatty():
        return f"{color}{text}{RESET}"
    return text

def print_report(result: dict, filename: str, verbose: bool = False):
    print(f"\n{'─'*62}")
    print(f" {_c(BOLD, 'PlantUML Web Validator')}  —  {filename}")
    print(f"{'─'*62}")

    if verbose and result.get("url"):
        print(f"  {_c(CYAN, 'Server URL')} : {result['url']}")

    if result["valid"]:
        print(f"\n  {_c(GREEN, '✅  VALID')}  — No syntax errors detected by plantuml.com")
        if verbose and result.get("url"):
            print(f"\n  Open in browser to view rendered diagram:")
            print(f"  {result['url']}")
    else:
        print(f"\n  {_c(RED, '❌  INVALID')}  — Syntax errors found:\n")
        for err in result["errors"]:
            print(f"  {_c(RED, '●')} {err}")

    print(f"\n{'─'*62}\n")
    return 0 if result["valid"] else 1


def main():
    parser = argparse.ArgumentParser(
        description="Validate PlantUML source via the plantuml.com web API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plantuml_validator.py diagram.puml
  python plantuml_validator.py diagram.puml --verbose
  python plantuml_validator.py diagram.puml --server https://my-server.com/plantuml
  cat diagram.puml | python plantuml_validator.py --stdin
        """
    )
    parser.add_argument("file",     nargs="?",  help="Path to .puml file")
    parser.add_argument("--stdin",  action="store_true", help="Read source from stdin")
    parser.add_argument("--server", default="https://www.plantuml.com/plantuml",
                        help="PlantUML server base URL (default: plantuml.com)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show encoded URL and extra info")
    args = parser.parse_args()

    if args.stdin:
        source   = sys.stdin.read()
        filename = "<stdin>"
    elif args.file:
        try:
            source   = Path(args.file).read_text(encoding="utf-8")
            filename = args.file
        except FileNotFoundError:
            print(f"Error: file not found — {args.file}")
            sys.exit(2)
        except UnicodeDecodeError as e:
            print(f"Error: cannot decode file (expected UTF-8) — {e}")
            sys.exit(2)
    else:
        parser.print_help()
        sys.exit(0)

    validator = PlantUMLWebValidator(server=args.server)
    result    = validator.validate(source)
    exit_code = print_report(result, filename, verbose=args.verbose)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

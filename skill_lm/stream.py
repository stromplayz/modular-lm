"""Stream tiny slices of HUGE datasets (Hugging Face 40GB+) - never full downloads.

The problem: datasets like TinyStories (2GB), C4 (~300GB+), Wikipedia dumps
(~20GB) dwarf any small-model budget and cannot live on GitHub.

The trick: byte-level HTTP Range requests. A trained pack needs megabytes,
not gigabytes - so we cut K small random windows straight out of the remote
file, exactly like sampling pages from a book we never check out:

    probe(url)                  -> total size, no body
    sample_slices(url, ...)     -> yields (offset, text) windows
    remote_corpus(url, mb)      -> a few MB of text = one pack's training data

GitHub Actions is a perfect runner for this: clean egress + 7GB disk, and
what gets committed back is a <1MB pack, never the corpus.

    python -m skill_lm.stream --url <parquet/txt/jsonl.gz> --slices 8 --slice-mb 2
"""
from __future__ import annotations

import argparse
import gzip
import io
import random
import urllib.request

UA = {"User-Agent": "modular-lm-stream/0.3"}

# datasets that grow forever get a soft cap so one URL can't explode a run
MAX_WINDOW_MB = 32


# ---------------------------------------------------------------------- #
def probe_size(url: str) -> int:
    """Content-Length via HEAD (no body download)."""
    req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        cr = r.headers.get("Content-Range")          # "bytes 0-0/123456789"
        if cr and "/" in cr:
            return int(cr.split("/")[-1])
        cl = r.headers.get("Content-Length")
        return int(cl) if cl else 0


def http_slice(url: str, offset: int, length: int, timeout: int = 120) -> bytes:
    """Exactly `length` bytes at `offset` - the only data that ever moves."""
    end = offset + length - 1
    req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={offset}-{end}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _decode(raw: bytes) -> str:
    """Best-effort text decode; transparent gzip; UTF-8 fallback chain."""
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def sample_slices(url: str, n_slices: int, slice_mb: float = 2.0,
                  seed: int = 0, trim_fraction: float = 0.05):
    """Yield (offset_mb, text) random windows from a remote file.

    Window edges are trimmed inward so partial lines/gzip-stream seams at the
    boundaries of a slice don't leak garbage into training text.
    """
    total = probe_size(url)
    if total <= 0:
        raise RuntimeError(f"cannot probe size of {url}")
    win = min(int(slice_mb * 1_000_000), MAX_WINDOW_MB * 1_000_000)
    trim = int(win * trim_fraction)
    rng = random.Random(seed)
    for i in range(n_slices):
        off = rng.randint(0, max(0, total - win - 1))
        raw = http_slice(url, off, win)
        text = _decode(raw)[trim // 2: -trim // 2 or None]
        yield off, text


def remote_corpus(url: str, target_mb: float = 8.0, slice_mb: float = 2.0,
                  seed: int = 0) -> str:
    """`target_mb` of training text pulled as small ranged windows."""
    n = max(1, round(target_mb / slice_mb))
    parts = []
    got = 0.0
    for off, text in sample_slices(url, n, slice_mb, seed):
        parts.append(text)
        got += len(text) / 1_000_000
        if got >= target_mb:
            break
    return "\n\n".join(parts)


def hf_url(repo_id: str, filename: str, repo_type: str = "datasets",
           revision: str = "main") -> str:
    """https://huggingface.co/<type>/<repo>/resolve/<rev>/<file> (Range-friendly)."""
    return f"https://huggingface.co/{repo_type}/{repo_id}/resolve/{revision}/{filename}"


# ---------------------------------------------------------------------- #
def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="range-slice a remote corpus")
    p.add_argument("--url", required=True)
    p.add_argument("--slices", type=int, default=4)
    p.add_argument("--slice-mb", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="write joined text to file")
    args = p.parse_args(argv)

    size = probe_size(args.url)
    print(f"remote size: {size / 1e9:.2f} GB - downloading "
          f"{args.slices * args.slice_mb / 1e3:.0f}k KB total ({args.slices} slices)")
    texts = []
    for off, text in sample_slices(args.url, args.slices, args.slice_mb, args.seed):
        print(f"  slice @ {off / 1e9:6.2f} GB -> {len(text):,} chars | {text[:70]!r}...")
        texts.append(text)
    joined = "\n\n".join(texts)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(joined)
        print(f"saved -> {args.out} ({len(joined)/1e6:.1f} MB)")
    else:
        print(f"total {len(joined):,} chars (use --out to save)")


if __name__ == "__main__":
    main()

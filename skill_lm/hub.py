"""Pack Hub: fetch skill submodels on demand from GitHub, cache, deload.

The main model is tiny and never changes. Skill packs live in a GitHub repo
(the HUB - default `stromplayz/modular-lm`, directory `packs/`). At runtime:

    need a skill?  ->  is it local?  ->  is it cached?  ->  fetch from hub
                   ->  load into SkillMixer  ->  use  ->  deload (drop from RAM)

Nothing is pre-downloaded: you only pull the packs a conversation actually
needs. A 100-pack hub costs nothing until a pack is requested.

    from skill_lm.hub import Hub
    hub = Hub("stromplayz/modular-lm")           # GitHub = the shard store
    names = hub.list_remote()                     # browse available packs
    path = hub.fetch("knowledge")                 # download + cache
    mixer = hub.load_mixer(["qa", "knowledge"])   # co-load any subset
    ... after use ...
    mixer.drop_pack("knowledge")                  # deload from memory

CLI:
    python -m skill_lm.hub --list
    python -m skill_lm.hub --fetch knowledge
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request

from .frozen import TrunkModel, SkillMixer, SkillPack

DEFAULT_REPO = "stromplayz/modular-lm"
DEFAULT_BRANCH = "main"
DEFAULT_SUBDIR = "packs"
CACHE_ROOT = os.environ.get(
    "MODULAR_LM_CACHE", os.path.expanduser("~/.cache/modular-lm/packs"))
UA = {"User-Agent": "modular-lm-hub/0.3"}


def _http(url: str, timeout: int = 30) -> bytes:
    headers = dict(UA)
    # optional auth raises API rate limits (raw.githubusercontent needs none)
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token and "api.github.com" in url:
        headers["Authorization"] = f"token {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


class Hub:
    """Remote pack registry backed by a GitHub repo (works with any repo,
    any branch; also accepts full raw URLs)."""

    def __init__(self, repo: str = DEFAULT_REPO, branch: str = DEFAULT_BRANCH,
                 subdir: str = DEFAULT_SUBDIR, cache_root: str = CACHE_ROOT) -> None:
        self.repo = repo.strip("/")
        self.branch = branch
        self.subdir = subdir.strip("/")
        self.cache_root = cache_root
        os.makedirs(cache_root, exist_ok=True)

    # -- urls ------------------------------------------------------------ #
    def raw_url(self, name: str) -> str:
        if name.startswith("http://") or name.startswith("https://"):
            return name  # full URL override
        return (f"https://raw.githubusercontent.com/{self.repo}/"
                f"{self.branch}/{self.subdir}/{name}.pack")

    def local_path(self, name: str) -> str:
        return os.path.join(self.cache_root,
                            f"{self.repo.replace('/', '__')}__{name}.pack")

    # -- browse ----------------------------------------------------------- #
    def list_remote(self) -> list[str]:
        """Pack names available on the hub (GitHub contents API)."""
        api = (f"https://api.github.com/repos/{self.repo}/contents/{self.subdir}"
               f"?ref={self.branch}")
        data = json.loads(_http(api))
        return sorted(item["name"][:-5] for item in data
                      if item["name"].endswith(".pack"))

    def list_cached(self) -> list[str]:
        out = []
        pre = f"{self.repo.replace('/', '__')}__"
        for f in os.listdir(self.cache_root):
            if f.startswith(pre) and f.endswith(".pack"):
                out.append(f[len(pre):-5])
        return sorted(out)

    # -- fetch / cache ------------------------------------------------------ #
    def fetch(self, name: str, force: bool = False, verify: bool = True) -> str:
        """Download a pack to the local cache; returns local path.

        Search order: cache -> hub (raw.githubusercontent.com).
        `verify` runs torch.load once so a truncated download never reaches
        the runtime.
        """
        lp = self.local_path(name)
        if os.path.exists(lp) and not force:
            return lp
        url = self.raw_url(name)
        blob = _http(url, timeout=120)
        tmp = lp + ".part"
        with open(tmp, "wb") as f:
            f.write(blob)
        if verify:
            pack = SkillPack.load(tmp)  # raises on corruption
            if pack.name != name and not name.startswith("http"):
                os.remove(tmp)
                raise ValueError(f"hub pack {name!r} has internal name {pack.name!r}")
        os.replace(tmp, lp)
        return lp

    def fingerprint(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    # -- load into runtime ---------------------------------------------------- #
    def load_mixer(self, trunk_path: str, pack_names: list[str],
                   local_dirs: list[str] | None = None,
                   allow_remote: bool = True) -> SkillMixer:
        """Trunk + requested packs, co-loaded. Resolution per pack:
        local_dirs (e.g. repo packs/) -> cache -> hub fetch."""
        trunk = TrunkModel.load(trunk_path, frozen=True)
        mixer = SkillMixer(trunk)
        local_dirs = local_dirs or []
        for nm in pack_names:
            path = None
            for d in local_dirs:
                cand = os.path.join(d, f"{nm}.pack")
                if os.path.exists(cand):
                    path = cand
                    break
            if path is None and os.path.exists(self.local_path(nm)):
                path = self.local_path(nm)
            if path is None:
                if not allow_remote:
                    raise FileNotFoundError(f"pack {nm!r} not local and remote disabled")
                path = self.fetch(nm)
            mixer.add_pack_file(path)
        return mixer

    # -- cache management -------------------------------------------------------- #
    def clear_cache(self, name: str | None = None) -> int:
        targets = [self.local_path(name)] if name else [
            os.path.join(self.cache_root, f)
            for f in os.listdir(self.cache_root)]
        removed = 0
        for t in targets:
            if os.path.exists(t):
                os.remove(t)
                removed += 1
        return removed


# ---------------------------------------------------------------------- #
def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="modular-lm pack hub")
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--branch", default=DEFAULT_BRANCH)
    p.add_argument("--list", action="store_true", help="list remote + cached packs")
    p.add_argument("--fetch", nargs="*", default=[], help="download packs to cache")
    p.add_argument("--clear", action="store_true", help="clear cache")
    args = p.parse_args(argv)

    hub = Hub(args.repo, args.branch)
    if args.clear:
        print(f"cleared {hub.clear_cache()} cached pack(s)")
    if args.fetch:
        for nm in args.fetch:
            path = hub.fetch(nm)
            print(f"fetched {nm:<16} -> {path}  ({os.path.getsize(path)/1e3:.0f} KB, "
                  f"sha {hub.fingerprint(path)})")
    if args.list or not (args.fetch or args.clear):
        print(f"hub: {args.repo}@{args.branch}")
        try:
            remote = hub.list_remote()
            print(f"remote ({len(remote)}): {', '.join(remote)}")
        except Exception as exc:  # noqa: BLE001
            print(f"remote unavailable: {exc}")
        cached = hub.list_cached()
        print(f"cached ({len(cached)}): {', '.join(cached) or '-'}")


if __name__ == "__main__":
    main()

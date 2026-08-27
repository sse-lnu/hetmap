"""unixcoder_emb.py

Single pipeline producing two embedding granularities per project:

  1. file    — class header + fields + constructors only
               encoded with simple truncation (max 512 tokens, one forward pass)
  2. member  — each non-boilerplate method body individually
               encoded with chunk_mean (CHUNK_SIZE=256, STRIDE=128)

Each of (1) and (2) is saved in zero-shot and fine-tuned variants:
  data/unixcoder_emb/{Name}_file_emb.npy / _file_keys.txt          (frozen)
  data/unixcoder_emb/{Name}_file_ft_emb.npy / _file_ft_keys.txt    (fine-tuned)
  data/unixcoder_emb/{Name}_member_emb.npy / _member_keys.txt      (frozen)
  data/unixcoder_emb/{Name}_member_ft_emb.npy / _member_ft_keys.txt (fine-tuned)
(file_pkg / file_pkg_ft variants produced with --pkg-name flag)

Run (from the HetMap repo root, with the source repos available under --project-dir):
  python -m hetmap.node_features.unixcoder_emb --project SH-3D
  python -m hetmap.node_features.unixcoder_emb --project SH-3D --frozen-only
"""

import os, warnings, argparse, re
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ── Section 1: Config ──────────────────────────────────────────────────────────

from pathlib import Path
from typing import Dict, List

_HERE       = Path(__file__).resolve().parent
MODEL_PATH  = _HERE / "unixcoder" / "unixcoder-base"
OUTPUT_DIR  = Path("data") / "unixcoder_emb"   # relative to CWD (HetMap root)

CHUNK_SIZE     = 256
STRIDE         = 128
TRUNC_MAX      = 512
BATCH_SIZE     = 8
LR             = 2e-5
EPOCHS         = 2
FREEZE_LAYERS  = 8
TEMPERATURE    = 0.05
PAIRS_PER_FILE = 2
HOP2_PROB      = 1.0
TOPK_N2        = 8
SEED           = 42
PKG_NAME       = False   # True → include full folder path in file-level class text

PROJECTS: Dict[str, str] = {
    "Bash":      "bash",
    "HDF":       "hdf",
    "HDC":       "hdc",
    "SH-3D":     "sweetHome",
    "A.UML":     "argouml",
    "C.Img":     "commons",
    "Jabref":    "jabref",
    "Lucene":    "lucene",
    "TeamMates": "teammates",
    "Chrome":    "chromium",
}

_REPOS: Dict[str, str] = {
    "argouml":   "argouml",
    "commons":   "commons/verify-commons-imaging-1.0-alpha2/apache-src/commons-imaging-1.0-alpha2-src/src",
    "jabref":    "jabref/src/main/java",
    "lucene":    "lucene",
    "sweetHome": "SweetHome3D",
    "teammates": "teammates",
    "chromium":  None,   # chromium repo root is project_dir itself (passed via --project-dir)
    "bash":      "bash",
    "hdf":       "hdf",
    "hdc":       "hdc",
}


# ── Section 2: AST extraction (self-contained) ─────────────────────────────────

_BODY_TYPES        = {"block", "constructor_body", "class_body",
                      "interface_body", "enum_body", "record_body"}
_COMMENT_TYPES     = {"line_comment", "block_comment"}
_CLASS_TYPES       = {"class_declaration", "interface_declaration",
                      "enum_declaration", "record_declaration"}
_FIELD_TYPES       = {"field_declaration", "enum_constant"}
_BOILERPLATE_NAMES = {
    "tostring", "equals", "hashcode", "compareto", "clone",
    "finalize", "readobject", "writeobject",
}
_GETTER_RE = re.compile(r"^(get|is|has)[A-Z0-9_]")
_SETTER_RE = re.compile(r"^set[A-Z0-9_]")

_parser_cache: Dict[str, object] = {}


def _get_java_parser():
    if "java" in _parser_cache:
        return _parser_cache["java"]
    try:
        import tree_sitter_java as _ts
        from tree_sitter import Language, Parser
        p = Parser(Language(_ts.language()))
        _parser_cache["java"] = p
        return p
    except Exception:
        _parser_cache["java"] = None
        return None


def _node_text(node) -> str:
    return node.text.decode("utf-8", errors="ignore")


def _method_name(node) -> str:
    for child in node.children:
        if child.type == "identifier":
            return _node_text(child)
    return ""


def _method_body(node) -> str:
    for child in node.children:
        if child.type in _BODY_TYPES:
            return _node_text(child)
    return ""


def _is_boilerplate(name: str, body: str) -> bool:
    if name.lower() in _BOILERPLATE_NAMES:
        return True
    body_clean = re.sub(r"\s+", " ", body).strip(" {}")
    if (_GETTER_RE.match(name) or _SETTER_RE.match(name)) and len(body_clean) < 80:
        return True
    return False


def _sig_text(node) -> str:
    parts = []
    for child in node.children:
        if child.type in _BODY_TYPES:
            break
        if child.type in _COMMENT_TYPES:
            continue
        t = _node_text(child).strip()
        if t:
            parts.append(t)
    return " ".join(parts)


def _header_text(node) -> str:
    parts = []
    for child in node.children:
        if child.type in _BODY_TYPES:
            break
        if child.type in _COMMENT_TYPES:
            continue
        parts.append(_node_text(child).strip())
    return " ".join(p for p in parts if p)


def _walk_classes(node):
    if node.type in _CLASS_TYPES:
        yield node
    for child in node.children:
        yield from _walk_classes(child)


def extract_file_view(source: str, path_prefix: str = "") -> str:
    """Class header + fields + constructors only (no method bodies)."""
    parser = _get_java_parser()
    if parser is None:
        return (path_prefix + " [no parser]").strip()
    try:
        root = parser.parse(source.encode("utf-8", errors="replace")).root_node
    except Exception:
        return (path_prefix + " [parse error]").strip()

    lines: List[str] = []
    if path_prefix:
        lines.append(f"PATH: {path_prefix}")

    for cls_node in _walk_classes(root):
        lines.append(_header_text(cls_node) + " {")
        body_node = next((c for c in cls_node.children if c.type in _BODY_TYPES), None)
        if body_node is None:
            lines.append("}")
            continue
        for child in body_node.children:
            if child.type in _FIELD_TYPES:
                lines.append(f"  field: {_node_text(child).strip()}")
            elif child.type == "constructor_declaration":
                sig  = _sig_text(child)
                body = _method_body(child)
                body_short = body[:200].rstrip() + ("..." if len(body) > 200 else "")
                lines.append(f"  constructor: {sig}")
                lines.append(f"    {body_short}")
        lines.append("}")

    if not lines or (len(lines) == 1 and lines[0].startswith("PATH:")):
        return (path_prefix + " [empty]").strip()
    return "\n".join(lines)


def extract_methods_text(source: str) -> str:
    """All method declarations in the class concatenated as one text (for chunked encoding).

    This is everything in the class body after name + constructors — i.e. the
    behavioural part of the class. Returns empty string if no methods found.
    """
    parser = _get_java_parser()
    if parser is None:
        return ""
    try:
        root = parser.parse(source.encode("utf-8", errors="replace")).root_node
    except Exception:
        return ""

    parts: List[str] = []

    def _collect(node):
        if node.type == "method_declaration":
            parts.append(_node_text(node))
            return
        for child in node.children:
            _collect(child)

    _collect(root)
    return "\n\n".join(parts)


# ── Section 3: Encode helpers ──────────────────────────────────────────────────

import numpy as np
import torch
import torch.nn.functional as F


def _load_model():
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    mdl = AutoModel.from_pretrained(MODEL_PATH)
    for p in mdl.embeddings.parameters():
        p.requires_grad_(False)
    for i, layer in enumerate(mdl.encoder.layer):
        if i < FREEZE_LAYERS:
            for p in layer.parameters():
                p.requires_grad_(False)
    return tok, mdl


def _encode_truncated(texts: List[str], tok, mdl, device, grad: bool = False) -> torch.Tensor:
    """Truncate to TRUNC_MAX tokens, single CLS, L2-normalised. Processes in mini-batches."""
    all_cls: List[torch.Tensor] = []

    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            enc  = tok(batch, max_length=TRUNC_MAX, truncation=True,
                       padding="max_length", return_tensors="pt")
            ids  = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            cls  = F.normalize(mdl(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :], dim=-1)
            # keep on GPU when grad=True (fine-tuning needs gradients on device);
            # offload to CPU when grad=False (inference) to avoid OOM across large datasets
            all_cls.append(cls if grad else cls.detach().cpu())
            del ids, mask

    return torch.cat(all_cls, dim=0)


def _encode_chunked(texts: List[str], tok, mdl, device, grad: bool = False) -> torch.Tensor:
    """Overlapping chunks, mean CLS per text, L2-normalised. Processes chunks in mini-batches."""
    # tokenize all texts, collect chunks and per-text counts
    all_ids: List[torch.Tensor] = []
    all_masks: List[torch.Tensor] = []
    counts: List[int] = []
    for text in texts:
        enc = tok(text, max_length=CHUNK_SIZE, stride=STRIDE,
                  return_overflowing_tokens=True, truncation=True,
                  padding="max_length", return_tensors="pt")
        all_ids.append(enc["input_ids"])
        all_masks.append(enc["attention_mask"])
        counts.append(enc["input_ids"].shape[0])

    ids_cat  = torch.cat(all_ids,  dim=0)   # (total_chunks, CHUNK_SIZE)
    mask_cat = torch.cat(all_masks, dim=0)
    total_chunks = ids_cat.shape[0]

    # encode all chunks in mini-batches, keeping results on CPU
    cls_chunks: List[torch.Tensor] = []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for i in range(0, total_chunks, BATCH_SIZE):
            ids_b  = ids_cat[i : i + BATCH_SIZE].to(device)
            mask_b = mask_cat[i : i + BATCH_SIZE].to(device)
            cls_b  = mdl(input_ids=ids_b, attention_mask=mask_b).last_hidden_state[:, 0, :]
            cls_chunks.append(cls_b.cpu())
            del ids_b, mask_b, cls_b

    cls_all = torch.cat(cls_chunks, dim=0)   # (total_chunks, hidden)

    # mean-pool chunks back to per-text embeddings
    embs: List[torch.Tensor] = []
    start = 0
    for n in counts:
        embs.append(cls_all[start : start + n].mean(0))
        start += n

    return F.normalize(torch.stack(embs), dim=-1)


# ── Section 4: Fine-tune helper ────────────────────────────────────────────────

def _finetune(texts: List[str], dep_keys: List[str], deps_data, tok, mdl, device, tag: str) -> None:
    """Contrastive fine-tuning on file-level pairs (modifies model in-place)."""
    from hetmap.node_features.contrastive.pairs import build_pairs, build_pos_set, PairDatasetWithIdx
    from hetmap.node_features.contrastive.loss import info_nce_masked
    from torch.utils.data import DataLoader
    from torch.optim import AdamW

    torch.manual_seed(SEED)
    pairs, stats = build_pairs(dep_keys, deps_data, pairs_per_file=PAIRS_PER_FILE,
                               hop2_prob=HOP2_PROB, topk_n2=TOPK_N2, seed=SEED)
    print(f"  [{tag}] pairs={stats['pairs']}  no_1hop={stats['no_1hop']}  no_2hop={stats['no_2hop']}")
    if not pairs:
        print(f"  [{tag}] WARN: no pairs — skipping fine-tuning")
        return

    pos_set   = build_pos_set(dep_keys, deps_data)
    loader    = DataLoader(PairDatasetWithIdx(pairs, texts), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    optimizer = AdamW([p for p in mdl.parameters() if p.requires_grad], lr=LR)

    for epoch in range(EPOCHS):
        mdl.train()
        total, n = 0.0, 0
        for ta, tb, ia, ib in loader:
            optimizer.zero_grad()
            ea = _encode_truncated(list(ta), tok, mdl, device, grad=True)
            eb = _encode_truncated(list(tb), tok, mdl, device, grad=True)
            bs = len(ia)
            conn = torch.zeros(bs, bs, dtype=torch.bool)
            for i in range(bs):
                for j in range(bs):
                    if i != j:
                        a, b = ia[i].item(), ib[j].item()
                        if (min(a, b), max(a, b)) in pos_set:
                            conn[i, j] = True
            loss = info_nce_masked(ea, eb, TEMPERATURE, conn.to(device))
            loss.backward(); optimizer.step()
            total += loss.item(); n += 1
        print(f"  [{tag}] epoch {epoch + 1}/{EPOCHS}  loss={total / max(n, 1):.4f}")

    mdl.eval()


# ── Section 5: Per-project loop ────────────────────────────────────────────────

def _save(emb: torch.Tensor, keys: List[str], path_emb: Path, path_keys: Path) -> None:
    np.save(str(path_emb), emb.cpu().float().numpy())
    path_keys.write_text("\n".join(keys), encoding="utf-8")
    print(f"    saved {path_emb.name}  shape={tuple(emb.shape)}")


def run_project(hetmap_name: str, arc_key: str, device: str, project_dir: str,
                pkg_name: bool = False, frozen_only: bool = False) -> None:
    print(f"\n{'='*60}\n  {hetmap_name}  ({arc_key})\n{'='*60}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _ftag_ft       = "file_pkg_ft" if pkg_name else "file_ft"
    _ftag_zs       = "file_pkg"    if pkg_name else "file"
    p_file_ft_emb  = OUTPUT_DIR / f"{hetmap_name}_{_ftag_ft}_emb.npy"
    p_file_ft_keys = OUTPUT_DIR / f"{hetmap_name}_{_ftag_ft}_keys.txt"
    p_mem_ft_emb   = OUTPUT_DIR / f"{hetmap_name}_member_ft_emb.npy"
    p_mem_ft_keys  = OUTPUT_DIR / f"{hetmap_name}_member_ft_keys.txt"
    p_file_zs_emb  = OUTPUT_DIR / f"{hetmap_name}_{_ftag_zs}_emb.npy"
    p_file_zs_keys = OUTPUT_DIR / f"{hetmap_name}_{_ftag_zs}_keys.txt"
    p_mem_zs_emb   = OUTPUT_DIR / f"{hetmap_name}_member_emb.npy"
    p_mem_zs_keys  = OUTPUT_DIR / f"{hetmap_name}_member_keys.txt"

    if frozen_only:
        if all(p.exists() for p in [p_file_zs_emb, p_file_zs_keys, p_mem_zs_emb, p_mem_zs_keys]):
            print("  [SKIP] all frozen outputs already exist.")
            return
    elif all(p.exists() for p in [p_file_ft_emb, p_file_ft_keys, p_mem_ft_emb, p_mem_ft_keys,
                                   p_file_zs_emb, p_file_zs_keys, p_mem_zs_emb, p_mem_zs_keys]):
        print("  [SKIP] all outputs already exist.")
        return

    # ── load project ──────────────────────────────────────────────────────────
    from hetmap.node_features.data_loader import load_project
    try:
        df, deps_data = load_project(arc_key, project_dir=project_dir)
    except Exception as e:
        print(f"  [SKIP] load_project failed: {e}")
        return

    print(f"  {len(df)} files")

    repo_sub  = _REPOS.get(arc_key)
    repo_root = Path(project_dir) if repo_sub is None else Path(project_dir) / repo_sub
    file_dep_keys = df["dep_key"].tolist()
    src_cache: Dict[str, str] = {}

    # ── file-level texts: filename + class header + fields + constructors ─────
    file_texts: List[str] = []
    for dk in file_dep_keys:
        if dk not in src_cache:
            try:
                src_cache[dk] = (repo_root / dk).read_text(encoding="utf-8", errors="replace")
            except Exception:
                src_cache[dk] = ""
        src = src_cache[dk]
        prefix = str(Path(dk).with_suffix("")) if pkg_name else Path(dk).stem
        file_texts.append(extract_file_view(src, path_prefix=prefix) if src else prefix + " [missing]")

    # ── member-level texts: all methods concatenated per file (built once) ────
    mem_texts: List[str] = []
    for dk in file_dep_keys:
        src = src_cache.get(dk, "")
        text = extract_methods_text(src) if src else ""
        mem_texts.append(text if text else "[no methods]")

    # ── load model ────────────────────────────────────────────────────────────
    tok, mdl = _load_model()
    mdl.to(device).eval()

    # ── zero-shot (frozen) embeddings ─────────────────────────────────────────
    if not (p_file_zs_emb.exists() and p_file_zs_keys.exists()):
        print("  encoding file (frozen) ...")
        _save(_encode_truncated(file_texts, tok, mdl, device), file_dep_keys, p_file_zs_emb, p_file_zs_keys)
    else:
        print("  [skip] file_zs exists")

    if not (p_mem_zs_emb.exists() and p_mem_zs_keys.exists()):
        print("  encoding member (frozen) ...")
        _save(_encode_chunked(mem_texts, tok, mdl, device), file_dep_keys, p_mem_zs_emb, p_mem_zs_keys)
    else:
        print("  [skip] member_zs exists")

    # ── fine-tune on file-level pairs ─────────────────────────────────────────
    _need_ft = not (p_file_ft_emb.exists() and p_file_ft_keys.exists() and
                    p_mem_ft_emb.exists() and p_mem_ft_keys.exists())
    if frozen_only:
        print("  [skip] fine-tuning (--frozen-only)")
    elif _need_ft:
        print("  fine-tuning ...")
        _finetune(file_texts, file_dep_keys, deps_data, tok, mdl, device, hetmap_name)

        if not (p_file_ft_emb.exists() and p_file_ft_keys.exists()):
            print("  encoding file (fine-tuned) ...")
            _save(_encode_truncated(file_texts, tok, mdl, device), file_dep_keys, p_file_ft_emb, p_file_ft_keys)
        else:
            print("  [skip] file_ft exists")

        if not (p_mem_ft_emb.exists() and p_mem_ft_keys.exists()):
            print("  encoding member (fine-tuned) ...")
            _save(_encode_chunked(mem_texts, tok, mdl, device), file_dep_keys, p_mem_ft_emb, p_mem_ft_keys)
        else:
            print("  [skip] member_ft exists")
    else:
        print("  [skip] all ft outputs exist — skipping fine-tuning")

    mdl.cpu(); del mdl, tok
    torch.cuda.empty_cache()


# ── Section 6: Entry point ─────────────────────────────────────────────────────

def main() -> None:
    _default_project_dir = str(_HERE.parents[2] / "data" / "projects")

    ap = argparse.ArgumentParser(description="UniXcoder embeddings: file / member (zero-shot + fine-tuned).")
    ap.add_argument("--project", default=None,
                    help="HetMap project name, e.g. 'SH-3D'. Default: all.")
    ap.add_argument("--project-dir", default=_default_project_dir,
                    help="Directory containing source-code repos (e.g. ant/, SweetHome3D/, ...).")
    ap.add_argument("--pkg-name", action="store_true",
                    help="Include full folder path in file-level class text; saves as _file_pkg_ft_.")
    ap.add_argument("--frozen-only", action="store_true",
                    help="Generate only zero-shot (frozen) embeddings; skip fine-tuning entirely.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}\noutput: {OUTPUT_DIR.resolve()}\nproject-dir: {args.project_dir}\n")

    if args.project is None:
        projects = PROJECTS
    elif args.project in PROJECTS:
        projects = {args.project: PROJECTS[args.project]}
    else:
        print(f"Unknown project '{args.project}'. Available: {list(PROJECTS)}")
        return

    for name, key in projects.items():
        run_project(name, key, device, args.project_dir,
                    pkg_name=args.pkg_name, frozen_only=args.frozen_only)

    print("\nDone.")


if __name__ == "__main__":
    main()

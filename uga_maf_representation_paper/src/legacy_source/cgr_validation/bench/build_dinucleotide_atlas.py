#!/usr/bin/env python3
"""
Universal Atlas Builder — EXP-022
Generates genome-wide Outside-In context centroids for:
  - SBS: 32 canonical trinucleotides (C/T center)
  - DBS: 10 canonical dinucleotides (COSMIC convention: AC,AT,CC,CG,CT,GC,GT,TA,TC,TT)

Strand canonicalization is applied during census so both strands contribute.
Layout per entry: [Upstream_x(d), Upstream_y(d), Downstream_x(d), Downstream_y(d)]
"""
import json
import sys
from pathlib import Path
import numpy as np
from numba import njit, prange

REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO))

from uga_atlas import CHROM_TO_ACCESSION  # noqa: E402

DATA_DIR = REPO / "data"
FASTA_PATH = DATA_DIR / "GRCH37/GCF_000001405.13/GCF_000001405.13_GRCh37_genomic.fna"
FAI_PATH   = FASTA_PATH.parent / (FASTA_PATH.name + ".fai")
OUT_SBS    = REPO / "cgr_validation_results/research/data/EXP022_atlas_genome_wide_45mer_universal.json"
OUT_DBS    = REPO / "cgr_validation_results/research/data/dinucleotide_atlas_d45.json"

BASES = "ACGT"

# ── bit tables (UGA encoding) ─────────────────────────────────────────────────
CORNER_X = np.zeros(256, dtype=np.float64)
CORNER_Y = np.zeros(256, dtype=np.float64)
IS_VALID = np.zeros(256, dtype=np.uint8)
COMP_ARR = np.zeros(256, dtype=np.uint8)   # complement lookup by ASCII value

for b, (x, y), c in [
    ('A', (0.0, 1.0), 'T'), ('C', (1.0, 1.0), 'G'),
    ('G', (0.0, 0.0), 'C'), ('T', (1.0, 0.0), 'A'),
]:
    for ch in (b, b.lower()):
        CORNER_X[ord(ch)] = x
        CORNER_Y[ord(ch)] = y
        IS_VALID[ord(ch)] = 1
    COMP_ARR[ord(b)]        = ord(c)
    COMP_ARR[ord(b.lower())] = ord(c)
    COMP_ARR[ord(c)]        = ord(b)
    COMP_ARR[ord(c.lower())] = ord(b)

# ── SBS: 32 canonical trinucleotides (C or T at center) ──────────────────────
def _tri_revcomp(t):
    _c = {'A':'T','C':'G','G':'C','T':'A'}
    return ''.join(_c[b] for b in reversed(t))

TRI_ALL  = [a+b+c for a in BASES for b in BASES for c in BASES]
SBS_KEYS = sorted({t if t[1] in 'CT' else _tri_revcomp(t) for t in TRI_ALL})
SBS_MAP  = np.full((256, 256, 256), -1, dtype=np.int16)
SBS_NEED_FLIP = np.zeros(len(SBS_KEYS), dtype=np.uint8)  # unused by kernel — flip decided per-locus

for t in TRI_ALL:
    canon = t if t[1] in 'CT' else _tri_revcomp(t)
    idx   = SBS_KEYS.index(canon)
    SBS_MAP[ord(t[0]), ord(t[1]), ord(t[2])] = idx
    # lower-case variants
    for t2 in [t, t.lower(), t[0]+t[1]+t[2].lower(),
               t[0].lower()+t[1]+t[2], t[0].lower()+t[1].lower()+t[2]]:
        try:
            SBS_MAP[ord(t2[0]), ord(t2[1]), ord(t2[2])] = idx
        except Exception:
            pass

# ── DBS: 10 canonical dinucleotides (COSMIC SBS78 convention) ─────────────────
# Keep those where ref di-nuc starts with C or T (same logic as SBS center)
def _di_revcomp(d):
    _c = {'A':'T','C':'G','G':'C','T':'A'}
    return ''.join(_c[b] for b in reversed(d))

DI_ALL   = [a+b for a in BASES for b in BASES]
DBS_KEYS = sorted({d if d[0] in 'CT' else _di_revcomp(d) for d in DI_ALL})
DBS_MAP  = np.full((256, 256), -1, dtype=np.int16)

for d in DI_ALL:
    canon = d if d[0] in 'CT' else _di_revcomp(d)
    idx   = DBS_KEYS.index(canon)
    DBS_MAP[ord(d[0]), ord(d[1])] = idx
    DBS_MAP[ord(d[0].lower()), ord(d[1])]        = idx
    DBS_MAP[ord(d[0]),          ord(d[1].lower())] = idx
    DBS_MAP[ord(d[0].lower()), ord(d[1].lower())] = idx

# ── FASTA index ───────────────────────────────────────────────────────────────
def read_fai(fai_path):
    idx = {}
    with open(fai_path) as f:
        for line in f:
            p = line.strip().split()
            if p:
                idx[p[0]] = (int(p[1]), int(p[2]), int(p[3]), int(p[4]))
    return idx

def fetch_chrom(fasta_fh, fai_entry):
    """Read full chromosome sequence from open FASTA file handle."""
    ref_len, offset, line_bases, line_bytes = fai_entry
    fasta_fh.seek(offset)
    n_lines = (ref_len + line_bases - 1) // line_bases
    raw = fasta_fh.read(n_lines * line_bytes)
    seq = raw.decode("ascii", errors="ignore").replace("\n", "").replace("\r", "")
    return seq[:ref_len].upper()

# ── numba census kernel ───────────────────────────────────────────────────────
@njit(parallel=True, fastmath=True)
def process_chunk(seq_bytes, corner_x, corner_y, is_valid, comp_arr,
                  sbs_map, dbs_map, n_sbs, n_dbs, d):
    """
    Race-condition-free: each parallel chunk writes to its own sub-array,
    merged afterwards in serial Python.
    """
    L = len(seq_bytes)
    if L < 2 * d + 2:
        return (np.zeros((n_sbs, 4 * d), dtype=np.float64),
                np.zeros(n_sbs, dtype=np.int64),
                np.zeros((n_dbs, 4 * d), dtype=np.float64),
                np.zeros(n_dbs, dtype=np.int64))

    CHUNK = 500_000
    n_chunks = max(1, (L - 2 * d) // CHUNK)

    sbs_chunk_sums  = np.zeros((n_chunks, n_sbs, 4 * d), dtype=np.float64)
    sbs_chunk_cnts  = np.zeros((n_chunks, n_sbs),         dtype=np.int64)
    dbs_chunk_sums  = np.zeros((n_chunks, n_dbs, 4 * d), dtype=np.float64)
    dbs_chunk_cnts  = np.zeros((n_chunks, n_dbs),         dtype=np.int64)

    for c in prange(n_chunks):
        i_start = d + c * CHUNK
        i_end   = min(i_start + CHUNK, L - d - 1)

        for i in range(i_start, i_end):
            # ── SBS ──────────────────────────────────────────────────────────
            s_idx = sbs_map[seq_bytes[i-1], seq_bytes[i], seq_bytes[i+1]]
            if s_idx >= 0:
                # If center was A or G, we are on the minus strand of the canonical;
                # flip context so we accumulate into the C/T-center canonical.
                center = seq_bytes[i]
                flipped = (center == 65 or center == 71)  # A=65, G=71
                for k in range(d):
                    if not flipped:
                        bl = seq_bytes[i - (k + 1)]
                        br = seq_bytes[i + (k + 1)]
                    else:
                        # revcomp: upstream neighbor becomes comp of downstream on original
                        bl = comp_arr[seq_bytes[i + (k + 1)]]
                        br = comp_arr[seq_bytes[i - (k + 1)]]
                    if is_valid[bl]:
                        sbs_chunk_sums[c, s_idx, k]       += corner_x[bl]
                        sbs_chunk_sums[c, s_idx, d + k]   += corner_y[bl]
                    if is_valid[br]:
                        sbs_chunk_sums[c, s_idx, 2*d + k] += corner_x[br]
                        sbs_chunk_sums[c, s_idx, 3*d + k] += corner_y[br]
                sbs_chunk_cnts[c, s_idx] += 1

            # ── DBS ──────────────────────────────────────────────────────────
            if i + 1 < L:
                d_idx = dbs_map[seq_bytes[i], seq_bytes[i+1]]
                if d_idx >= 0:
                    first_base = seq_bytes[i]
                    flipped = (first_base == 65 or first_base == 71)  # not C or T
                    for k in range(d):
                        if not flipped:
                            bl = seq_bytes[i - (k + 1)]   if i-(k+1) >= 0 else 0
                            br = seq_bytes[i + 2 + k]      if i+2+k < L  else 0
                        else:
                            bl = comp_arr[seq_bytes[i + 2 + k]]      if i+2+k < L  else 0
                            br = comp_arr[seq_bytes[i - (k + 1)]]    if i-(k+1)>=0 else 0
                        if is_valid[bl]:
                            dbs_chunk_sums[c, d_idx, k]       += corner_x[bl]
                            dbs_chunk_sums[c, d_idx, d + k]   += corner_y[bl]
                        if is_valid[br]:
                            dbs_chunk_sums[c, d_idx, 2*d + k] += corner_x[br]
                            dbs_chunk_sums[c, d_idx, 3*d + k] += corner_y[br]
                    dbs_chunk_cnts[c, d_idx] += 1

    # Merge chunks (serial)
    sbs_sums = sbs_chunk_sums.sum(axis=0)
    sbs_cnts = sbs_chunk_cnts.sum(axis=0)
    dbs_sums = dbs_chunk_sums.sum(axis=0)
    dbs_cnts = dbs_chunk_cnts.sum(axis=0)
    return sbs_sums, sbs_cnts, dbs_sums, dbs_cnts


def run():
    MAX_D   = 45
    n_sbs   = len(SBS_KEYS)
    n_dbs   = len(DBS_KEYS)

    print(f"SBS canonical keys ({n_sbs}): {SBS_KEYS}")
    print(f"DBS canonical keys ({n_dbs}): {DBS_KEYS}")

    total_sbs_sums = np.zeros((n_sbs, 4 * MAX_D), dtype=np.float64)
    total_sbs_cnts = np.zeros(n_sbs, dtype=np.int64)
    total_dbs_sums = np.zeros((n_dbs, 4 * MAX_D), dtype=np.float64)
    total_dbs_cnts = np.zeros(n_dbs, dtype=np.int64)

    fai = read_fai(FAI_PATH)

    with open(FASTA_PATH, "rb") as fasta_fh:
        for chrom, accession in CHROM_TO_ACCESSION.items():
            if chrom == "MT":
                continue
            if accession not in fai:
                print(f"  Chromosome {chrom} ({accession}): not in FAI, skipping.")
                continue
            print(f"  Processing chromosome {chrom} ({accession})...", end=" ", flush=True)
            seq = fetch_chrom(fasta_fh, fai[accession])
            seq_bytes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
            s_sums, s_cnts, d_sums, d_cnts = process_chunk(
                seq_bytes, CORNER_X, CORNER_Y, IS_VALID, COMP_ARR,
                SBS_MAP, DBS_MAP, n_sbs, n_dbs, MAX_D
            )
            total_sbs_sums += s_sums
            total_sbs_cnts += s_cnts
            total_dbs_sums += d_sums
            total_dbs_cnts += d_cnts
            print(f"SBS counts: {s_cnts.sum():,}  DBS counts: {d_cnts.sum():,}")

    # ── Write SBS atlas ──────────────────────────────────────────────────────
    sbs_atlas = {}
    for i, key in enumerate(SBS_KEYS):
        if total_sbs_cnts[i] > 0:
            sbs_atlas[key] = (total_sbs_sums[i] / total_sbs_cnts[i]).tolist()
        else:
            print(f"  WARNING: no counts for SBS key {key}")
    with open(OUT_SBS, "w") as f:
        json.dump(sbs_atlas, f)
    print(f"\nSBS atlas saved -> {OUT_SBS}  ({len(sbs_atlas)} keys)")

    # ── Write DBS atlas ──────────────────────────────────────────────────────
    dbs_atlas = {}
    for i, key in enumerate(DBS_KEYS):
        if total_dbs_cnts[i] > 0:
            dbs_atlas[key] = (total_dbs_sums[i] / total_dbs_cnts[i]).tolist()
        else:
            print(f"  WARNING: no counts for DBS key {key}")
    with open(OUT_DBS, "w") as f:
        json.dump(dbs_atlas, f)
    print(f"DBS atlas saved -> {OUT_DBS}  ({len(dbs_atlas)} keys)")


if __name__ == "__main__":
    run()

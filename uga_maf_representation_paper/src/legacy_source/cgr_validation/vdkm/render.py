import numpy as np
import math

# Mapping ASCII chars to bit values for UGA
X_MAP = {'A': 0.0, 'C': 1.0, 'G': 0.0, 'T': 1.0, 'N': 0.0}
Y_MAP = {'A': 1.0, 'C': 1.0, 'G': 0.0, 'T': 0.0, 'N': 0.0}
BASE = "ACGT"
COMP = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
XY_TO_BASE = {(X_MAP[b], Y_MAP[b]): b for b in BASE}
PAYLOAD_SCHEMA_LEGACY = "legacy"
PAYLOAD_SCHEMA_MASKED = "masked"
PAYLOAD_SCHEMA_LENGTH = "length"

CHROM_TO_ACCESSION = {
    "1": "NC_000001.10", "2": "NC_000002.11", "3": "NC_000003.11", "4": "NC_000004.11",
    "5": "NC_000005.9", "6": "NC_000006.11", "7": "NC_000007.13", "8": "NC_000008.10",
    "9": "NC_000009.11", "10": "NC_000010.10", "11": "NC_000011.9", "12": "NC_000012.11",
    "13": "NC_000013.10", "14": "NC_000014.8", "15": "NC_000015.9", "16": "NC_000016.9",
    "17": "NC_000017.10", "18": "NC_000018.9", "19": "NC_000019.9", "20": "NC_000020.10",
    "21": "NC_000021.8", "22": "NC_000022.10", "X": "NC_000023.10", "Y": "NC_000024.9",
    "MT": "NC_012920.1"
}

from pathlib import Path

class FastaReader:
    def __init__(self, fasta_path: Path):
        self.fasta_path = Path(fasta_path)
        self.fai_path = self.fasta_path.parent / (self.fasta_path.name + ".fai")
        self.index = {}
        self._f_fasta = None
        
        if self.fai_path.exists():
            with open(self.fai_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        self.index[parts[0]] = (int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
                        
    def open(self):
        if self._f_fasta is None and self.fasta_path.exists():
            self._f_fasta = open(self.fasta_path, "rb")
            
    def close(self):
        if self._f_fasta is not None:
            self._f_fasta.close()
            self._f_fasta = None
            
    def fetch(self, chrom: str, pos1: int, window_len: int = 45) -> str:
        self.open()
        if self._f_fasta is None:
            return "N" * window_len
            
        accession = CHROM_TO_ACCESSION.get(str(chrom).upper().replace("CHR", ""), None)
        if accession is None:
            accession = str(chrom)
            
        if accession not in self.index:
            return "N" * window_len
            
        ref_len, offset, line_bases, line_bytes = self.index[accession]
        
        start0 = pos1 - 1 - (window_len // 2)
        # If window_len is -1, fetch to end
        if window_len == -1:
            end0 = ref_len
            bases_to_read = end0 - start0
        else:
            end0 = start0 + window_len
            bases_to_read = window_len
        
        prefix_ns, suffix_ns = 0, 0
        if start0 < 0:
            prefix_ns = abs(start0)
            start0 = 0
        if end0 > ref_len:
            suffix_ns = end0 - ref_len
            end0 = ref_len
            
        if start0 >= end0:
            return "N" * (window_len if window_len > 0 else 0)
            
        bases_to_read = end0 - start0
        lines_before = start0 // line_bases
        bytes_before = lines_before * line_bytes + (start0 % line_bases)
        byte_start = offset + bytes_before
        
        bytes_to_read = bases_to_read + (bases_to_read // line_bases + 2) * (line_bytes - line_bases)
        
        self._f_fasta.seek(byte_start)
        raw_data = self._f_fasta.read(bytes_to_read)
        
        seq = raw_data.decode("ascii", errors="ignore").replace("\n", "").replace("\r", "")
        seq = seq[:bases_to_read]
        return ("N" * prefix_ns) + seq.upper() + ("N" * suffix_ns)

    def fetch_range(self, chrom: str, start1: int, end1: int) -> str:
        """Fetch an inclusive 1-based interval, padding out-of-bound bases with N."""
        start1 = int(start1)
        end1 = int(end1)
        if end1 < start1:
            return ""
        self.open()
        width = end1 - start1 + 1
        if self._f_fasta is None:
            return "N" * width

        accession = CHROM_TO_ACCESSION.get(str(chrom).upper().replace("CHR", ""), None)
        if accession is None:
            accession = str(chrom)

        if accession not in self.index:
            return "N" * width

        ref_len, offset, line_bases, line_bytes = self.index[accession]
        start0 = start1 - 1
        end0 = end1
        prefix_ns, suffix_ns = 0, 0
        if start0 < 0:
            prefix_ns = -start0
            start0 = 0
        if end0 > ref_len:
            suffix_ns = end0 - ref_len
            end0 = ref_len
        if start0 >= end0:
            return "N" * width

        bases_to_read = end0 - start0
        lines_before = start0 // line_bases
        bytes_before = lines_before * line_bytes + (start0 % line_bases)
        byte_start = offset + bytes_before
        bytes_to_read = bases_to_read + (bases_to_read // line_bases + 2) * (line_bytes - line_bases)

        self._f_fasta.seek(byte_start)
        raw_data = self._f_fasta.read(bytes_to_read)
        seq = raw_data.decode("ascii", errors="ignore").replace("\n", "").replace("\r", "")
        seq = seq[:bases_to_read]
        return ("N" * prefix_ns) + seq.upper() + ("N" * suffix_ns)

def revcomp(seq):
    return "".join(COMP.get(b, 'N') for b in reversed(seq.upper()))

def coord_to_bits(val, d=13):
    """Vectorized float-to-bits. Not used in optimized binary flow, but kept for legacy benchmark support."""
    v = np.atleast_1d(val).astype(np.float64)
    bits = np.zeros((len(v), int(d)), dtype=np.float64)
    for i in range(int(d)):
        v *= 2.0
        mask = (v >= 1.0)
        bits[mask, i] = 1.0
        v[mask] -= 1.0
    return bits.flatten() if len(v) == 1 else bits

def bits_to_coord(bits):
    """Convert a dyadic bit prefix back to its [0, 1) coordinate."""
    arr = np.asarray(bits, dtype=np.float64).ravel()
    weights = 2.0 ** -np.arange(1, len(arr) + 1, dtype=np.float64)
    return float(np.dot(arr, weights))

def cgr_encode(seq, weight=0.5):
    """Legacy-compatible direct UGA coordinate encoder.

    The current UGA implementation stores the dyadic bits directly. This helper
    is kept for older diagnostics that compare bit extraction against a scalar
    coordinate path.
    """
    _ = weight
    seq = str(seq or "").upper()
    xb, yb = fast_seq_to_bits(seq, len(seq))
    x = bits_to_coord(xb)
    y = bits_to_coord(yb)
    return x, y, x, y, True

def cgr_decode(x, y, length):
    """Decode a direct UGA coordinate prefix into a DNA string."""
    xb = coord_to_bits(x, int(length))
    yb = coord_to_bits(y, int(length))
    chars = []
    ok = True
    for bx, by in zip(xb, yb):
        base = XY_TO_BASE.get((float(bx), float(by)))
        if base is None:
            chars.append("N")
            ok = False
        else:
            chars.append(base)
    return "".join(chars), ok

def fast_seq_to_bits(seq, d):
    """ Directly converts a sequence to UGA bit vectors. """
    xb = np.array([X_MAP.get(b, 0.0) for b in seq[:d]], dtype=np.float64)
    yb = np.array([Y_MAP.get(b, 0.0) for b in seq[:d]], dtype=np.float64)
    if len(xb) < d:
        pad = np.zeros(d - len(xb), dtype=np.float64)
        xb = np.concatenate([xb, pad])
        yb = np.concatenate([yb, pad])
    return xb, yb

def encode_bicgr_context(left_seq, right_seq=None, d=13):
    """ Optimized Binary Clean-Context Encoder. Walks outward from the variant. """
    if right_seq is None:
        seq = str(left_seq).upper()
        half = len(seq) // 2
        left_seq = seq[:half]
        right_seq = seq[half + 1:]
    up_walk = str(left_seq).upper()[::-1] 
    down_walk = str(right_seq).upper()
    xl_bits, yl_bits = fast_seq_to_bits(up_walk, d)
    xr_bits, yr_bits = fast_seq_to_bits(down_walk, d)
    return np.concatenate([xl_bits, yl_bits, xr_bits, yr_bits])

def normalize_payload_schema(payload_schema):
    schema = str(payload_schema or PAYLOAD_SCHEMA_LEGACY).lower().replace("-", "_")
    aliases = {
        "bits": PAYLOAD_SCHEMA_LEGACY,
        "zero": PAYLOAD_SCHEMA_LEGACY,
        "legacy_zero": PAYLOAD_SCHEMA_LEGACY,
        "presence": PAYLOAD_SCHEMA_MASKED,
        "presence_mask": PAYLOAD_SCHEMA_MASKED,
        "mask": PAYLOAD_SCHEMA_MASKED,
        "block_length": PAYLOAD_SCHEMA_LENGTH,
        "length_code": PAYLOAD_SCHEMA_LENGTH,
        "length_coded": PAYLOAD_SCHEMA_LENGTH,
    }
    schema = aliases.get(schema, schema)
    if schema not in {PAYLOAD_SCHEMA_LEGACY, PAYLOAD_SCHEMA_MASKED, PAYLOAD_SCHEMA_LENGTH}:
        raise ValueError(f"Unsupported payload schema: {payload_schema}")
    return schema

def payload_length_bit_count(d_payload):
    """Bits needed for payload lengths 0..d plus an overflow state."""
    return int(math.ceil(math.log2(int(d_payload) + 2)))

def payload_block_dim(d_payload, payload_schema=PAYLOAD_SCHEMA_LEGACY):
    """Number of features used by one REF or ALT payload block."""
    schema = normalize_payload_schema(payload_schema)
    d_payload = int(d_payload)
    if schema == PAYLOAD_SCHEMA_MASKED:
        return 3 * d_payload
    if schema == PAYLOAD_SCHEMA_LENGTH:
        return 2 * d_payload + payload_length_bit_count(d_payload)
    return 2 * d_payload

def universal_vector_dim(d_context, d_payload, payload_schema=PAYLOAD_SCHEMA_LEGACY):
    """Total UGA vector width for a context/payload configuration."""
    return 4 * int(d_context) + 2 * payload_block_dim(d_payload, payload_schema)

def encode_alt_cgr_walk(seq, d=13, payload_schema=PAYLOAD_SCHEMA_LEGACY):
    """Optimized payload walk.

    legacy: [X, Y] with zero padding.
    masked: [X, Y, M] where M marks real payload bases, so absent slots cannot
    collide with G=(0,0) or with right-padded G bases.
    length: [X, Y, L] where L is a block-level binary length code for compact
    prefix payloads, reserving code d+1 for overflow/truncation.
    """
    schema = normalize_payload_schema(payload_schema)
    seq = str(seq or "").upper()
    xb, yb = fast_seq_to_bits(seq, d)
    if schema == PAYLOAD_SCHEMA_LEGACY:
        return np.concatenate([xb, yb])
    if schema == PAYLOAD_SCHEMA_LENGTH:
        n = min(len(seq), int(d))
        length_code = int(d) + 1 if len(seq) > int(d) else n
        return np.concatenate([xb, yb, int_to_bits(length_code, payload_length_bit_count(d))])
    mask = np.zeros(int(d), dtype=np.float64)
    mask[: min(len(seq), int(d))] = 1.0
    return np.concatenate([xb, yb, mask])

def int_to_bits(value, width):
    width = int(width)
    value = int(value)
    return np.array([(value >> (width - 1 - i)) & 1 for i in range(width)], dtype=np.float64)

def bits_to_int(bits):
    out = 0
    for bit in np.asarray(bits, dtype=np.float64).ravel():
        out = (out << 1) | int(bit >= 0.5)
    return out

def decode_masked_payload(bits, d):
    """Decode a masked payload block created by encode_alt_cgr_walk(..., masked)."""
    arr = np.asarray(bits, dtype=np.float64)
    d = int(d)
    if len(arr) != 3 * d:
        raise ValueError(f"Masked payload block must have length {3 * d}, got {len(arr)}")
    xb, yb, mask = arr[:d], arr[d:2 * d], arr[2 * d:3 * d]
    chars = []
    ok = True
    for bx, by, present in zip(xb, yb, mask):
        if present <= 0.5:
            continue
        base = XY_TO_BASE.get((float(bx), float(by)))
        if base is None:
            chars.append("N")
            ok = False
        else:
            chars.append(base)
    return "".join(chars), ok

def decode_length_payload(bits, d):
    """Decode a length-coded payload block.

    Returns (sequence, ok, overflow). If overflow is True, the sequence is the
    stored d-base prefix and the original payload was longer than d.
    """
    arr = np.asarray(bits, dtype=np.float64)
    d = int(d)
    width = payload_length_bit_count(d)
    expected = 2 * d + width
    if len(arr) != expected:
        raise ValueError(f"Length-coded payload block must have length {expected}, got {len(arr)}")
    xb, yb, lb = arr[:d], arr[d:2 * d], arr[2 * d:]
    code = bits_to_int(lb)
    overflow = code == d + 1
    ok = code <= d + 1
    length = d if overflow else min(code, d)
    chars = []
    for bx, by in zip(xb[:length], yb[:length]):
        base = XY_TO_BASE.get((float(bx), float(by)))
        if base is None:
            chars.append("N")
            ok = False
        else:
            chars.append(base)
    return "".join(chars), ok, overflow

def trim_shared_alleles(ref, alt):
    r, a = (ref or "").upper(), (alt or "").upper()
    i = 0
    while i < len(r) and i < len(a) and r[i] == a[i]:
        i += 1
    r, a = r[i:], a[i:]
    while len(r) > 0 and len(a) > 0 and r[-1] == a[-1]:
        r, a = r[:-1], a[:-1]
    return r, a

def encode_variant_universal(
    ref_context_45mer,
    alt_sequence,
    d=13,
    *,
    d_context=None,
    d_payload=None,
    ref_allele=None,
    canonicalize=True,
    payload_schema=PAYLOAD_SCHEMA_LEGACY,
):
    dc = int(d_context if d_context is not None else d)
    dp = int(d_payload if d_payload is not None else d)
    seq = str(ref_context_45mer).upper()
    alt = str(alt_sequence or "").upper()
    
    half = len(seq) // 2
    if ref_allele is None:
        ref_allele = seq[half]
    ref = str(ref_allele).upper()
    
    if canonicalize and ref in ("A", "G"):
        seq = revcomp(seq)
        alt = "".join(COMP.get(b, "N") for b in reversed(alt))
        ref = COMP.get(ref, "N")

    ref_start = seq.find(ref, half - len(ref), half + len(ref))
    if ref_start == -1: ref_start = half
    
    left_context = seq[:ref_start]
    right_context = seq[ref_start + len(ref):]
    rp, ap = trim_shared_alleles(ref, alt)
    xl_bits, yl_bits = fast_seq_to_bits(left_context[::-1], dc)
    xr_bits, yr_bits = fast_seq_to_bits(right_context, dc)
    ref_bits = encode_alt_cgr_walk(rp, d=dp, payload_schema=payload_schema)
    alt_bits = encode_alt_cgr_walk(ap, d=dp, payload_schema=payload_schema)
    
    return np.concatenate([
        xl_bits, yl_bits,
        ref_bits,
        xr_bits, yr_bits,
        alt_bits
    ])

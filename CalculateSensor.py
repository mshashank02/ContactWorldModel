from typing import Dict, List, Tuple
import math

# ------------------ apportionment helpers ------------------

def _largest_remainder(values: List[float], labels: List[str], total_int: int, priority: List[str] = None) -> Dict[str, int]:
    """Largest Remainder (Hamilton) apportionment with optional tie-break priority."""
    floors = [math.floor(v) for v in values]
    result = {lab: f for lab, f in zip(labels, floors)}
    leftover = total_int - sum(floors)
    if leftover <= 0:
        return result

    rema = [v - f for v, f in zip(values, floors)]
    pr_rank = {lab: i for i, lab in enumerate(priority)} if priority else {}

    order = list(range(len(labels)))
    order.sort(key=lambda i: (-rema[i], pr_rank.get(labels[i], 10**9), labels[i]))
    for i in order[:leftover]:
        result[labels[i]] += 1
    return result


# ------------------ group totals (area-weighted; extras -> tips) ------------------

def _allocate_groups_int(Ap: float, Apx: float, At: float, Ntotal: int, Rppx: float, Rpt: float) -> Dict[str, int]:
    if min(Ap, Apx, At) <= 0 or min(Rppx, Rpt) <= 0 or Ntotal <= 0:
        raise ValueError("Areas, ratios, and Ntotal must be positive.")
    Dp = Ntotal / (Ap + (Apx / Rppx) + (At / Rpt))
    Np_cont  = Dp * Ap
    Npx_cont = (Dp / Rppx) * Apx
    Nt_cont  = (Dp / Rpt) * At
    labels = ["Np", "Npx", "Nt"]
    vals   = [Np_cont, Npx_cont, Nt_cont]
    prio   = ["Nt", "Npx", "Np"]  # bias extras to tips
    out = _largest_remainder(vals, labels, Ntotal, priority=prio)
    assert sum(out.values()) == Ntotal
    return out


def _split_palm_area_weighted(Np: int, Ap1: float, Ap2: float) -> Dict[str, int]:
    if Ap1 <= 0 or Ap2 <= 0:
        raise ValueError("Ap1 and Ap2 must be positive.")
    Palm_cont  = Np * (Ap1 / (Ap1 + Ap2))
    LMeta_cont = Np * (Ap2 / (Ap1 + Ap2))
    return _largest_remainder(
        [Palm_cont, LMeta_cont],
        ["TS_palm", "TS_lfmetacarpal"],
        Np,
        priority=["TS_palm", "TS_lfmetacarpal"]
    )


# ------------------ scalable naming (seed existing, then auto-extend) ------------------

# Seed (existing) site names you already have — used first
SEEDS: Dict[str, List[str]] = {
    "TS_palm": [
        "robot0:T_palm_b0","robot0:T_palm_bl","robot0:T_palm_bm","robot0:T_palm_br",
        "robot0:T_palm_fl","robot0:T_palm_fm","robot0:T_palm_fr","robot0:T_palm_b1",
    ],
    "TS_lfmetacarpal": ["robot0:T_lfmetacarpal_front"],

    "TS_ffproximal": [
        "robot0:T_ffproximal_front_left_bottom","robot0:T_ffproximal_front_right_bottom",
        "robot0:T_ffproximal_front_left_top","robot0:T_ffproximal_front_right_top",
        "robot0:T_ffproximal_back_left","robot0:T_ffproximal_back_right",
        "robot0:T_ffproximal_tip",
    ],
    "TS_ffmiddle": [
        "robot0:T_ffmiddle_front_left","robot0:T_ffmiddle_front_right",
        "robot0:T_ffmiddle_back_left","robot0:T_ffmiddle_back_right",
        "robot0:T_ffmiddle_tip",
    ],
    "TS_fftip": [
        "robot0:T_fftip_front_left","robot0:T_fftip_front_right",
        "robot0:T_fftip_back_left","robot0:T_fftip_back_right","robot0:T_fftip_tip",
    ],

    "TS_mfproximal": [
        "robot0:T_mfproximal_front_left_bottom","robot0:T_mfproximal_front_right_bottom",
        "robot0:T_mfproximal_front_left_top","robot0:T_mfproximal_front_right_top",
        "robot0:T_mfproximal_back_left","robot0:T_mfproximal_back_right",
        "robot0:T_mfproximal_tip",
    ],
    "TS_mfmiddle": [
        "robot0:T_mfmiddle_front_left","robot0:T_mfmiddle_front_right",
        "robot0:T_mfmiddle_back_left","robot0:T_mfmiddle_back_right",
        "robot0:T_mfmiddle_tip",
    ],
    "TS_mftip": [
        "robot0:T_mftip_front_left","robot0:T_mftip_front_right",
        "robot0:T_mftip_back_left","robot0:T_mftip_back_right","robot0:T_mftip_tip",
    ],

    "TS_rfproximal": [
        "robot0:T_rfproximal_front_left_bottom","robot0:T_rfproximal_front_right_bottom",
        "robot0:T_rfproximal_front_left_top","robot0:T_rfproximal_front_right_top",
        "robot0:T_rfproximal_back_left","robot0:T_rfproximal_back_right",
        "robot0:T_rfproximal_tip",
    ],
    "TS_rfmiddle": [
        "robot0:T_rfmiddle_front_left","robot0:T_rfmiddle_front_right",
        "robot0:T_rfmiddle_back_left","robot0:T_rfmiddle_back_right",
        "robot0:T_rfmiddle_tip",
    ],
    "TS_rftip": [
        "robot0:T_rftip_front_left","robot0:T_rftip_front_right",
        "robot0:T_rftip_back_left","robot0:T_rftip_back_right","robot0:T_rftip_tip",
    ],

    "TS_lfproximal": [
        "robot0:T_lfproximal_front_left_bottom","robot0:T_lfproximal_front_right_bottom",
        "robot0:T_lfproximal_front_left_top","robot0:T_lfproximal_front_right_top",
        "robot0:T_lfproximal_back_left","robot0:T_lfproximal_back_right",
        "robot0:T_lfproximal_tip",
    ],
    "TS_lfmiddle": [
        "robot0:T_lfmiddle_front_left","robot0:T_lfmiddle_front_right",
        "robot0:T_lfmiddle_back_left","robot0:T_lfmiddle_back_right",
        "robot0:T_lfmiddle_tip",
    ],
    "TS_lftip": [
        "robot0:T_lftip_front_left","robot0:T_lftip_front_right",
        "robot0:T_lftip_back_left","robot0:T_lftip_back_right","robot0:T_lftip_tip",
    ],

    "TS_thproximal": [
        "robot0:T_thproximal_front_left","robot0:T_thproximal_front_right",
        "robot0:T_thproximal_back_left","robot0:T_thproximal_back_right",
        "robot0:T_thproximal_tip",
    ],
    "TS_thmiddle": [
        "robot0:T_thmiddle_front_left","robot0:T_thmiddle_front_right",
        "robot0:T_thmiddle_back_left","robot0:T_thmiddle_back_right",
        "robot0:T_thmiddle_tip",
    ],
    "TS_thtip": [
        "robot0:T_thtip_front_left","robot0:T_thtip_front_right",
        "robot0:T_thtip_back_left","robot0:T_thtip_back_right","robot0:T_thtip_tip",
    ],
}

# Auto-extend with numbered site names if allocation > seed count
PREFIX: Dict[str, str] = {
    "TS_palm": "robot0:T_palm_auto",
    "TS_lfmetacarpal": "robot0:T_lfmetacarpal_auto",
    "TS_ffproximal": "robot0:T_ffproximal_auto",
    "TS_ffmiddle": "robot0:T_ffmiddle_auto",
    "TS_fftip": "robot0:T_fftip_auto",
    "TS_mfproximal": "robot0:T_mfproximal_auto",
    "TS_mfmiddle": "robot0:T_mfmiddle_auto",
    "TS_mftip": "robot0:T_mftip_auto",
    "TS_rfproximal": "robot0:T_rfproximal_auto",
    "TS_rfmiddle": "robot0:T_rfmiddle_auto",
    "TS_rftip": "robot0:T_rftip_auto",
    "TS_lfproximal": "robot0:T_lfproximal_auto",
    "TS_lfmiddle": "robot0:T_lfmiddle_auto",
    "TS_lftip": "robot0:T_lftip_auto",
    "TS_thproximal": "robot0:T_thproximal_auto",
    "TS_thmiddle": "robot0:T_thmiddle_auto",
    "TS_thtip": "robot0:T_thtip_auto",
}

PHALANX_KEYS = [
    "TS_ffproximal","TS_ffmiddle",
    "TS_mfproximal","TS_mfmiddle",
    "TS_rfproximal","TS_rfmiddle",
    "TS_lfproximal","TS_lfmiddle",
    "TS_thproximal","TS_thmiddle",
]
TIP_KEYS = ["TS_fftip","TS_mftip","TS_rftip","TS_lftip","TS_thtip"]


def _names_for_region(region: str, count: int) -> List[Tuple[str, str]]:
    """
    Return (touch_name, site_name) pairs:
    - Use seed sites first (exact names you already have)
    - Then auto-extend with numbered names (no positions yet)
    Touch name is derived from site name by T_ -> TS_.
    """
    pairs: List[Tuple[str, str]] = []
    seeds = SEEDS.get(region, [])
    take = min(count, len(seeds))
    for s in seeds[:take]:
        touch = s.replace("robot0:T_", "robot0:TS_")
        pairs.append((touch, s))

    remaining = count - take
    if remaining > 0:
        base = PREFIX[region]
        for i in range(1, remaining + 1):
            site_name = f"{base}_{i:03d}"
            touch_name = site_name.replace("robot0:T_", "robot0:TS_")
            pairs.append((touch_name, site_name))
    return pairs


# ------------------ main builder ------------------

def build_sensor_xml_scaled(
    Ap: float, Apx: float, At: float,
    Ntotal: int, Rppx: float, Rpt: float,
    Ap1: float, Ap2: float
) -> Tuple[str, Dict[str, int]]:
    """
    Returns:
      - XML string with <mujoco><sensor>...</sensor></mujoco>
      - stats dict (group totals + per-region counts)
    """
    # Group totals
    groups = _allocate_groups_int(Ap, Apx, At, Ntotal, Rppx, Rpt)
    Np, Npx, Nt = groups["Np"], groups["Npx"], groups["Nt"]

    # Palm split (area-weighted)
    palm = _split_palm_area_weighted(Np, Ap1, Ap2)

    # Equal splits across phalanx & tips (integerized)
    phal = _largest_remainder([Npx/len(PHALANX_KEYS)]*len(PHALANX_KEYS), PHALANX_KEYS, Npx)
    tips = _largest_remainder([Nt/len(TIP_KEYS)]*len(TIP_KEYS), TIP_KEYS, Nt)

    # Desired counts per region
    desired: Dict[str, int] = {}
    desired.update(palm)
    desired.update(phal)
    desired.update(tips)

    # Emit in sections — NOTE: lfmetacarpal ONLY appears in PALM section now.
    sections = [
        ("PALM", ["TS_palm", "TS_lfmetacarpal"]),
        ("FOREFINGER", ["TS_ffproximal","TS_ffmiddle","TS_fftip"]),
        ("MIDDLE FINGER", ["TS_mfproximal","TS_mfmiddle","TS_mftip"]),
        ("RING FINGER", ["TS_rfproximal","TS_rfmiddle","TS_rftip"]),
        ("LITTLE FINGER", ["TS_lfproximal","TS_lfmiddle","TS_lftip"]),
        ("THUMB", ["TS_thproximal","TS_thmiddle","TS_thtip"]),
    ]

    lines = ['<mujoco>', '    <sensor>']
    for title, keys in sections:
        lines.append(f'\n        <!--{title}-->')
        for k in keys:
            n = desired.get(k, 0)
            if n <= 0:
                continue
            for touch_name, site_name in _names_for_region(k, n):
                lines.append(f'        <touch name="{touch_name}" site="{site_name}"></touch>')
    lines += ['\n    </sensor>', '</mujoco>']
    xml = "\n".join(lines)

    # Stats
    stats: Dict[str, int] = {"Np": Np, "Npx": Npx, "Nt": Nt, "Ntotal": Ntotal}
    for k in ["TS_palm", "TS_lfmetacarpal"] + PHALANX_KEYS + TIP_KEYS:
        stats[k] = desired.get(k, 0)
    stats["check_sum"] = sum(stats[k] for k in ["TS_palm", "TS_lfmetacarpal"] + PHALANX_KEYS + TIP_KEYS)

    return xml, stats


# ------------------ example run ------------------

if __name__ == "__main__":
    xml, stats = build_sensor_xml_scaled(
        Ap=1.0, Apx=1.0, At=1.0,
        Ntotal=92, Rppx=1.0, Rpt=1.0,
        Ap1=25.0, Ap2=25.0
    )

    # Print XML
    print(xml)

    # Print stats summary
    print("\n# ---- Allocation Stats ----")
    print(f"Group totals: Np={stats['Np']}, Npx={stats['Npx']}, Nt={stats['Nt']}, Ntotal={stats['Ntotal']}")
    print("Palm split:   TS_palm={:d}, TS_lfmetacarpal={:d}".format(stats["TS_palm"], stats["TS_lfmetacarpal"]))
    print("Phalanxes:    " + ", ".join(f"{k}={stats[k]}" for k in PHALANX_KEYS))
    print("Tips:         " + ", ".join(f"{k}={stats[k]}" for k in TIP_KEYS))
    ok = "OK" if stats["check_sum"] == stats["Ntotal"] else f"!! MISMATCH ({stats['check_sum']} != {stats['Ntotal']})"
    print(f"Total check:  {stats['check_sum']} == Ntotal? {ok}")

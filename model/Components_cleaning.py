import pandas as pd

def load_data(bridges_path: str, roads_path: str):
    """Load bridges (Excel) and roads (CSV) datasets."""
    bridges = pd.read_excel(bridges_path)
    roads = pd.read_csv(roads_path)
    return bridges, roads

def filter_road(bridges: pd.DataFrame, roads: pd.DataFrame, road_name: str):
    """Return only rows belonging to the selected road."""
    bridges_r = bridges[bridges["road"] == road_name].copy()
    roads_r = roads[roads["road"] == road_name].copy()
    return bridges_r, roads_r

def prepare_road_links(roads_r: pd.DataFrame):
    """
    Convert chainage points into link segments with start_km and end_km.
    """
    roads_r = roads_r.sort_values("chainage").reset_index(drop=True)

    roads_r["start_km"] = roads_r["chainage"]
    roads_r["end_km"] = roads_r["chainage"].shift(-1)

    # Remove last row (no next chainage)
    roads_r = roads_r[roads_r["road"] == roads_r["road"].shift(-1)]

    roads_r["model_type"] = "link"
    roads_r["length"] = roads_r["end_km"] - roads_r["start_km"]

    return roads_r[[
        "road", "chainage", "model_type", "name", "lat", "lon",
        "length", "start_km", "end_km"]]


def prepare_bridges(bridges_r: pd.DataFrame):
    """
    Convert bridge lengths (meters) into km and compute start/end positions.
    """
    bridges_r = bridges_r.copy()
    bridges_r["length_km"] = bridges_r["length"] / 1000
    bridges_r["start_km"] = bridges_r["chainage"] - bridges_r["length_km"] / 2
    bridges_r["end_km"] = bridges_r["chainage"] + bridges_r["length_km"] / 2
    bridges_r["model_type"] = "bridge"
    bridges_r["length"] = bridges_r["length_km"]

    bridges_r = bridges_r.rename(columns={"LRPName": "lrp"})

    return bridges_r[[
        "road", "model_type", "name", "lat", "lon",
        "length", "condition", "start_km", "end_km"
    ]]


def split_links_at_bridges(roads_r: pd.DataFrame, bridges_r: pd.DataFrame):
    """
    Split road links wherever a bridge starts or ends.
    Remove segments that fall inside a bridge.
    """
    cut_points = sorted(
        set(bridges_r["start_km"].tolist() + bridges_r["end_km"].tolist()))

    bridge_intervals = [(row.start_km, row.end_km) for _, row in bridges_r.iterrows()]

    def inside_bridge(s, e):
        """Check if a segment lies fully inside any bridge interval."""
        for bs, be in bridge_intervals:
            if s >= bs and e <= be:
                return True
        return False

    split_links = []

    for _, row in roads_r.iterrows():
        s, e = row["start_km"], row["end_km"]

        # Find cut points inside this link
        internal_cuts = [x for x in cut_points if s < x < e]
        boundaries = [s] + internal_cuts + [e]

        # Create new segments
        for i in range(len(boundaries) - 1):
            seg_start = boundaries[i]
            seg_end = boundaries[i + 1]

            if inside_bridge(seg_start, seg_end):
                continue

            new_row = row.copy()
            new_row["start_km"] = seg_start
            new_row["end_km"] = seg_end
            new_row["length"] = seg_end - seg_start

            split_links.append(new_row)

    return pd.DataFrame(split_links)

def build_full_network(roads_r: pd.DataFrame, bridges_r: pd.DataFrame):
    """
    Combine split road links and bridges into a single ordered network.
    Add source/sink nodes and assign unique IDs.
    """
    combined = pd.concat([roads_r, bridges_r], ignore_index=True)
    combined = combined.sort_values("start_km").reset_index(drop=True)

    # Convert first and last link to source/sink
    if combined.loc[0, "model_type"] == "link":
        combined.loc[0, "model_type"] = "source"
    if combined.loc[combined.shape[0]-1, "model_type"] == "link":
        combined.loc[combined.shape[0]-1, "model_type"] = "sink"

    combined["id"] = range(1_000_000, 1_000_000 + len(combined))

    # Convert km → meters
    combined["length"] = combined["length"] * 1000

    return combined[["id", "road", "model_type", "name","lat", "lon", "length", "condition",
        "start_km", "end_km"]]

def process_road_network(bridges_path, roads_path, road_name="N1"):
    """Full pipeline to generate the processed road network for a given road."""
    bridges, roads = load_data(bridges_path, roads_path)
    bridges_r, roads_r = filter_road(bridges, roads, road_name)

    roads_r = prepare_road_links(roads_r)
    bridges_r = prepare_bridges(bridges_r)

    roads_r = split_links_at_bridges(roads_r, bridges_r)
    full = build_full_network(roads_r, bridges_r)

    return full

full_N1 = process_road_network("../data/BMMS_overview.xlsx", "../data/_roads3.csv", "N1")
full_N1.to_csv("../data/N1_AS2.csv", index=False)


"""Project the catalog into two dimensions for the Song Map tab.

Audio features are standardized, then reduced to three PCA components, so
the map can be drawn in 2D (the first two) or 3D (all three). The preference
profile only stores energy and valence. For the other features, the user
marker uses the average of the songs retrieved for that profile, so it lands
among the songs the scorer judged closest. With no retrieved songs it falls
back to the catalog average.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = {
    "energy": "energy",
    "valence": "valence",
    "danceability": "danceability",
    "acousticness": "acousticness",
    "instrumentalness": "instrumental",
    "tempo_bpm": "tempo",
}

FEATURES = (
    "energy",
    "valence",
    "danceability",
    "acousticness",
    "instrumentalness",
    "tempo_bpm",
)


def _matrix(songs: Sequence[Dict]) -> np.ndarray:
    rows = []
    for song in songs:
        rows.append([float(song[feature]) for feature in FEATURES])
    return np.asarray(rows, dtype=float)


def project_songs(songs: Sequence[Dict]) -> Dict:
    """Fit PCA on the catalog and return coordinates plus the transform."""
    usable = [song for song in songs if all(song.get(feature) is not None for feature in FEATURES)]
    if len(usable) < 2:
        return {"points": [], "means": {}, "scale": [], "components": [], "pca_mean": [], "explained": [], "axis_labels": []}
    matrix = _matrix(usable)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    n_components = min(3, len(usable), len(FEATURES))
    pca = PCA(n_components=n_components, random_state=0)
    coords = pca.fit_transform(scaled)
    if n_components < 3:
        coords = np.hstack([coords, np.zeros((len(usable), 3 - n_components))])
    points = []
    for song, xy in zip(usable, coords):
        points.append({
            "id": song["id"],
            "title": song["title"],
            "artist": song["artist"],
            "genre": song["genre"],
            "mood": song["mood"],
            "x": float(xy[0]),
            "y": float(xy[1]),
            "z": float(xy[2]),
        })
    means = {feature: float(matrix[:, index].mean()) for index, feature in enumerate(FEATURES)}
    return {
        "points": points,
        "means": means,
        "scale": [float(value) for value in scaler.scale_],
        "components": [[float(value) for value in row] for row in pca.components_],
        "pca_mean": [float(value) for value in pca.mean_],
        "explained": [float(value) for value in pca.explained_variance_ratio_],
        "axis_labels": [axis_label(row) for row in pca.components_],
    }


def axis_label(component: Sequence[float], top: int = 2) -> str:
    """Name a PCA axis by its strongest features, e.g. "energy ↑ · acousticness ↓"."""
    ranked = sorted(zip(component, FEATURES), key=lambda item: -abs(item[0]))[:top]
    return " · ".join(f"{FEATURE_NAMES[feature]} {'↑' if weight > 0 else '↓'}" for weight, feature in ranked)


def project_profile(
    profile: Optional[Dict],
    model: Dict,
    reference_songs: Optional[Sequence[Dict]] = None,
) -> Optional[List[float]]:
    """Place the parsed profile in the same PCA space.

    Energy and valence come from the profile. The other features come from
    reference_songs (usually the retrieved songs), or the catalog mean.
    """
    if not profile or not model.get("components"):
        return None
    means = model["means"]
    fill = dict(means)
    usable = [
        song for song in reference_songs or []
        if all(song.get(feature) is not None for feature in FEATURES)
    ]
    if usable:
        for feature in FEATURES:
            fill[feature] = sum(float(song[feature]) for song in usable) / len(usable)
    vector = np.array([
        float(profile.get(feature, fill[feature])) if feature in ("energy", "valence") else fill[feature]
        for feature in FEATURES
    ], dtype=float)
    scale = np.array(model["scale"], dtype=float)
    scale[scale == 0] = 1.0
    # Same steps as StandardScaler, then PCA's own centering.
    scaled = (vector - np.array([means[feature] for feature in FEATURES])) / scale
    scaled = scaled - np.array(model["pca_mean"], dtype=float)
    components = np.array(model["components"], dtype=float)
    coords = [float(value) for value in components @ scaled]
    return coords + [0.0] * (3 - len(coords))

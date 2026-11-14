"""Project the catalog into two dimensions for the Song Map tab.

Audio features are standardized, then reduced with PCA. The preference
profile only stores energy and valence, so the other axes of the user marker
use the catalog average.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
        return {"points": [], "means": {}, "scale": [], "components": [], "pca_mean": []}
    matrix = _matrix(usable)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(scaled)
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
        })
    means = {feature: float(matrix[:, index].mean()) for index, feature in enumerate(FEATURES)}
    return {
        "points": points,
        "means": means,
        "scale": [float(value) for value in scaler.scale_],
        "components": [[float(value) for value in row] for row in pca.components_],
        "pca_mean": [float(value) for value in pca.mean_],
    }


def project_profile(profile: Optional[Dict], model: Dict) -> Optional[List[float]]:
    """Place the parsed profile in the same PCA space."""
    if not profile or not model.get("components"):
        return None
    means = model["means"]
    vector = np.array([
        float(profile.get("energy", means["energy"])),
        float(profile.get("valence", means["valence"])),
        means["danceability"],
        means["acousticness"],
        means["instrumentalness"],
        means["tempo_bpm"],
    ], dtype=float)
    scale = np.array(model["scale"], dtype=float)
    scale[scale == 0] = 1.0
    centered = (vector - np.array([means[feature] for feature in FEATURES])) / scale
    # StandardScaler also subtracts its own mean, which is the same feature mean.
    scaled = centered
    scaled = scaled - np.array(model["pca_mean"], dtype=float)
    components = np.array(model["components"], dtype=float)
    return [float(value) for value in components @ scaled]

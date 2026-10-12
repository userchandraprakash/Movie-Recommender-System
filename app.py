from __future__ import annotations

import ast
from pathlib import Path
from zipfile import BadZipFile, ZipFile
from typing import Any

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

BASE_DIR = Path(__file__).resolve().parent
MOVIES_FILE = "tmdb_5000_movies.csv"
CREDITS_FILE = "tmdb_5000_credits.csv"

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1200px;}
      .hero {padding: 1.4rem 1.5rem; border-radius: 18px; background: linear-gradient(120deg, rgba(90, 90, 115, .16), rgba(110, 160, 175, .12)); margin-bottom: 1.2rem;}
      .eyebrow {font-size: .85rem; opacity: .75; text-transform: uppercase; letter-spacing: .08em;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Discover your next favorite</div>
      <h1 style="margin:.35rem 0">🎬 Movie Recommender</h1>
      <p style="margin-bottom:0">Choose a movie and get five recommendations based on its story, genres, keywords, cast, and director.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def find_csv(filename: str) -> Path | None:
    """Locate an extracted dataset CSV in common project folders."""
    for folder in ("Data", "data", ""):
        candidate = BASE_DIR / folder / filename
        if candidate.is_file():
            return candidate
    return None


def read_dataset_csv(filename: str) -> pd.DataFrame:
    """Read a CSV directly or from the compressed dataset archive."""
    csv_path = find_csv(filename)
    if csv_path is not None:
        return pd.read_csv(csv_path)

    archives = (
        BASE_DIR / "Data" / "tmdb_dataset.zip",
        BASE_DIR / "data" / "tmdb_dataset.zip",
        BASE_DIR / "tmdb_dataset.zip",
    )
    for archive_path in archives:
        if not archive_path.is_file():
            continue
        try:
            with ZipFile(archive_path) as archive:
                if filename in archive.namelist():
                    with archive.open(filename) as file_obj:
                        return pd.read_csv(file_obj)
        except (BadZipFile, OSError) as exc:
            raise ValueError(f"Could not read dataset archive: {archive_path.name}") from exc

    raise FileNotFoundError(
        f"Could not find {filename}. Place both CSV files in Data/ or data/, "
        "or add Data/tmdb_dataset.zip containing both CSV files."
    )


def parse_records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        result = ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError):
        return []
    return result if isinstance(result, list) else []


def names_from_column(value: Any) -> list[str]:
    names: list[str] = []
    for item in parse_records(value):
        if isinstance(item, dict) and item.get("name"):
            names.append(str(item["name"]))
    return names


def top_cast(value: Any, limit: int = 3) -> list[str]:
    return names_from_column(value)[:limit]


def director_name(value: Any) -> list[str]:
    result = []
    for item in parse_records(value):
        if isinstance(item, dict) and item.get("job") == "Director" and item.get("name"):
            result.append(str(item["name"]))
            break
    return result


def pretty_genres(value: Any) -> list[str]:
    return names_from_column(value)


@st.cache_resource(show_spinner="Loading the movie dataset and preparing recommendations...")
def load_recommender() -> tuple[pd.DataFrame, Any, NearestNeighbors]:
    try:
        movies_df = read_dataset_csv(MOVIES_FILE)
        credits_df = read_dataset_csv(CREDITS_FILE)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Movie data is missing. Add Data/tmdb_dataset.zip (included in this project) "
            "or place both TMDB CSV files directly inside Data/."
        ) from exc
    movies_df.columns = movies_df.columns.str.strip()
    credits_df.columns = credits_df.columns.str.strip()

    required_movie = {"id", "title", "overview", "genres", "keywords"}
    required_credit = {"movie_id", "cast", "crew"}
    missing_movie = sorted(required_movie - set(movies_df.columns))
    missing_credit = sorted(required_credit - set(credits_df.columns))
    if missing_movie or missing_credit:
        details = []
        if missing_movie:
            details.append("movies CSV missing: " + ", ".join(missing_movie))
        if missing_credit:
            details.append("credits CSV missing: " + ", ".join(missing_credit))
        raise ValueError("The uploaded CSV files do not match the expected TMDB dataset. " + " | ".join(details))

    movies_df["id"] = pd.to_numeric(movies_df["id"], errors="coerce")
    credits_df["movie_id"] = pd.to_numeric(credits_df["movie_id"], errors="coerce")
    credits_df = credits_df.drop_duplicates(subset="movie_id")
    merged = movies_df.merge(
        credits_df[["movie_id", "cast", "crew"]],
        left_on="id",
        right_on="movie_id",
        how="inner",
        validate="one_to_one",
    )

    data = merged[["title", "overview", "genres", "keywords", "cast", "crew"]].copy()
    data = data.dropna(subset=["title", "overview"])
    data["title"] = data["title"].astype(str).str.strip()
    data["overview"] = data["overview"].astype(str)
    data = data[data["title"].ne("")].reset_index(drop=True)

    tags = []
    for row in data.itertuples(index=False):
        title, overview, genres, keywords, cast, crew = row
        terms = (
            overview.split()
            + names_from_column(genres)
            + names_from_column(keywords)
            + top_cast(cast)
            + director_name(crew)
        )
        tags.append(" ".join(str(term) for term in terms))
    data["tags"] = tags
    data = data[data["tags"].str.strip().ne("")].reset_index(drop=True)

    if len(data) < 2:
        raise ValueError("The dataset does not contain enough valid movies to create recommendations.")

    vectorizer = CountVectorizer(max_features=5000, stop_words="english", dtype="float32")
    vectors = vectorizer.fit_transform(data["tags"])
    model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=min(6, len(data)))
    model.fit(vectors)

    display_data = data[["title", "overview", "genres"]].copy()
    display_data["genre_list"] = display_data["genres"].apply(pretty_genres)
    return display_data.reset_index(drop=True), vectors, model


try:
    movies, movie_vectors, recommender = load_recommender()
except Exception as exc:
    st.error("The movie recommender could not start.")
    st.write(str(exc))
    st.info(
        "Check that Data/tmdb_dataset.zip is included in the GitHub repository, "
        "or place both TMDB CSV files directly inside Data/."
    )
    st.stop()

st.caption(f"Dataset ready · {len(movies):,} movies available")

movie_titles = movies["title"].tolist()
selected_index = st.selectbox(
    "Select a movie",
    options=range(len(movie_titles)),
    format_func=lambda idx: movie_titles[idx],
)

if st.button("Show Recommendations", type="primary", use_container_width=True):
    # Query the selected movie from the fitted sparse feature matrix.
    distances, indices = recommender.kneighbors(
        movie_vectors[selected_index],
        n_neighbors=min(6, len(movies)),
        return_distance=True,
    )
    recommended_indices = [int(idx) for idx in indices[0] if int(idx) != selected_index][:5]

    if not recommended_indices:
        st.warning("No recommendations were found for this movie.")
    else:
        st.subheader(f"Movies similar to {movies.iloc[selected_index]['title']}")
        cards = st.columns(5)
        for position, (column, movie_index) in enumerate(zip(cards, recommended_indices), start=1):
            movie = movies.iloc[movie_index]
            with column:
                with st.container(border=True):
                    st.caption(f"RECOMMENDATION {position}")
                    st.markdown(f"**{movie['title']}**")
                    genres = movie["genre_list"]
                    if genres:
                        st.caption(" · ".join(genres[:3]))
                    overview = str(movie["overview"]).strip()
                    if overview and overview.lower() != "nan":
                        st.write(overview if len(overview) <= 360 else overview[:357].rsplit(" ", 1)[0] + "...")
                    else:
                        st.write("No description is available for this movie.")

st.divider()
st.caption("Recommendations are generated from the TMDB 5000 movie metadata dataset. This demo does not display movie posters.")

# Movie Recommender System

🚀 **[Open Live Website](PASTE-YOUR-LIVE-URL-HERE)**

A Movie Recommendation System built with Python,
Pandas, Scikit-learn, and Streamlit.

An English-language Streamlit app that recommends five similar movies based on story overview, genres, keywords, cast, and director. **Movie posters and all poster-related functionality have been removed.**

## Included files

- `app.py` — Streamlit application
- `requirements.txt` — Python dependencies
- `Data/tmdb_dataset.zip` — both TMDB CSV datasets in one compressed file

The app reads the compressed archive directly. **Do not extract the archive or rename it** before uploading through GitHub's website.

## Run locally

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open the local URL shown in the terminal, usually `http://localhost:8501`.

## Deploy free on Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload `app.py`, `requirements.txt`, `README.md`, `.gitignore`, and `Data/tmdb_dataset.zip`.
3. Open <https://share.streamlit.io/> and sign in with GitHub.
4. Create an app from your repository, select branch `main`, and set the main file path to `app.py`.
5. Click **Deploy**.

Do not upload your local `.venv` folder. The app reads either the included `Data/tmdb_dataset.zip` or both CSV files placed directly inside `Data/`.

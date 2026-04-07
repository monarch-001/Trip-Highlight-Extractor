# Trip Highlight Extractor

AI-powered Streamlit app that automatically organizes your trip photos into **events**, **people**, **scenery categories**, and a **highlights** folder.

Live demo (Streamlit): https://trip-highlight-extractor-h9soue3rflvqwh9dttmxzp.streamli.app

---

## What it does

Given a folder of images (JPG/PNG) from a trip, the app will:

- **Filter low-quality photos** (too blurry / too dark)
- **Detect faces** and classify images into:
  - **Solo** (1 face)
  - **Group** (2+ faces)
  - **Scenery** (0 faces) with labels like *Nature / City / Food / Monument*
- **Cluster photos into events** using photo timestamps (EXIF) and a configurable time gap
- **Cluster people** across photos using **InsightFace** face embeddings, with optional within-event refinement using clothing/body similarity
- **Remove near-duplicates** inside each event using CLIP visual embeddings + cosine similarity
- **Export an organized folder structure**, plus a **Highlights** folder containing top photos per event (by aesthetic score)

---

## How it works (high level)

- **Face detection + face embeddings:** `insightface` (RetinaFace + ArcFace)
- **Image embeddings + aesthetic score + scenery classification:** OpenAI **CLIP** (`openai/clip-vit-base-patch32`) via `transformers`
- **Clustering:** `DBSCAN` for both event grouping (time-based) and person grouping (embedding distance)

Key files:

- `app.py` — Streamlit UI (local folder / Google Drive input, controls, preview, naming, export)
- `engine.py` — core pipeline (batch analysis, scoring, clustering, redundancy filtering, export)
- `drive_utils.py` — Google Drive folder download helper
- `requirements.txt` — Python dependencies

---

## Running locally

### 1) Clone

```bash
git clone https://github.com/monarch-001/Trip-Highlight-Extractor.git
cd Trip-Highlight-Extractor
```

### 2) Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Start the Streamlit app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

---

## Using the app

1. Choose **Input Source**:
   - **Local Folder**: paste a path containing images
   - **Google Drive**: paste a folder ID (or full folder URL)
2. Configure settings:
   - **Event Gap (Minutes)**: time gap used to split events (default 60)
   - **Redundancy Sensitivity**: similarity threshold for deduping (default 0.92)
3. Click **Start Organizing**.
4. Review stats/preview, optionally **name people**.
5. Click **Select Folder & Export Now** to write the organized output.

---

## Output folder structure

Exports a structure like:

```text
<export_dir>/
  By_Event/
    Event_0/
      Solo_Pics/
        Person_1/
        Person_2/
      Group_Photos/
      Scenery/
        Nature/
        City/
        Food/
        Monument/
    Event_1/
      ...
  People/
    Person_1/
    Person_2/
  Highlights/
    Event_0_<photo>.jpg
    Event_1_<photo>.jpg
```

---

## Google Drive setup (optional)

If you want to use **Google Drive** as the input source:

1. Create a Google Cloud project and enable the **Google Drive API**.
2. Create a **Service Account**, and download its JSON key.
3. Save the key as `credentials.json` in the project root.
4. Share the target Drive folder with the service account email.

> Note: `credentials.json` should not be committed to the repository.

---

## Notes / limitations

- Best results with photos that include EXIF timestamps.
- First run may take time while models download (InsightFace + CLIP).
- GPU is used automatically if available; otherwise it falls back to CPU.

---

## License

No license file is currently included in this repository. Add a `LICENSE` if you plan to distribute this project.
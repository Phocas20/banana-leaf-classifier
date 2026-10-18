# Banana Leaf Disease Classifier — deployment package

This project wraps the supplied `Model.keras` in a small Streamlit web application.

## Model details from the training notebook
- Task: banana-leaf image classification
- Classes (in output order): `cordana`, `sigatoka`, `pestalotiopsis`, `healthy`
- Input: RGB image resized to 224 × 224
- Pixel scaling before prediction: divide pixel values by 255 (0–1)
- Architecture described in the notebook: MobileNetV2 transfer learning

## Run locally
Use Python 3.11 if possible.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown in the terminal (usually http://localhost:8501).

## Deploy online with Streamlit Community Cloud
1. Sign in to GitHub and create a **new public repository** named `banana-leaf-classifier`.
2. Upload these three files to the repository root:
   - `app.py`
   - `requirements.txt`
   - `Model.keras`
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Select **Create app**, choose your repository and branch, and set the main file path to `app.py`.
5. Click **Deploy** and wait for dependencies to install. The first build can take several minutes.
6. Open the generated `streamlit.app` URL and test it with a leaf image.

## Important
- This package is a deployment starter, not proof that a public deployment has succeeded.
- The model file is about 24 MB, so it should fit in a normal GitHub repository.
- If deployment fails due to dependency/runtime compatibility, check the Streamlit app logs. Do not change the class order unless the model's training notebook is also changed.
- Predictions depend on the training data and test performance. Do not use this tool alone to make high-impact crop-treatment decisions.

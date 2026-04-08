# Escáner de Documentos A4 — versión híbrida C++ + Streamlit


This version keeps the interface in **Streamlit/Python** and moves the scanning core to **C++** using **pybind11**.

## What has been moved to C++

- Automatic document detection.
- Perspective transformation.
- Quadrilateral expansion.
- Final black frame trimming.
- Manual scanning based on 4 points.

## What remains in Python

- Streamlit interface.
- Image upload.
- Manual click preview.
- Final result download.

## Structure

```text
A4-Document-Scanner-CPP/
├── app.py
├── CMakeLists.txt
├── pyproject.toml
├── requirements.txt
├── packages.txt
└── cpp/
    └── docscanner.cpp
```

## Local execution

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deployment on Streamlit Community Cloud

Upload the entire project to GitHub and select `app.py` as the main file.

### Important

This version **needs to compile a C++ module** during the build process. For that reason, the repository includes:

- `requirements.txt` for Python dependencies.
- `packages.txt` for system dependencies.
- `pyproject.toml` and `CMakeLists.txt` to build `docscanner_cpp`.

## Realistic note

This approach is technically correct for using C++ inside Streamlit, but the most delicate part of deployment is compiling OpenCV C++ in the cloud. If the build on Streamlit becomes too heavy or fails because of system dependencies, the more stable alternative is to return to a 100% Python version for deployment and keep C++ for a native app or a separate backend.

Built By Alan Masoud


# A4 Document Scanner - Streamlit Cloud Safe Version

This version is designed to run directly on Streamlit Community Cloud without compiling a local C++ module.

## Files
- `app.py`
- `requirements.txt`

## Deploy
1. Upload these files to your GitHub repository.
2. In Streamlit Community Cloud, select `app.py` as the entrypoint.
3. Reboot or redeploy the app.

## Why this version
Your previous hybrid version tried to import `docscanner_cpp`. If that extension is not built during deployment, the app will show `No module named 'docscanner_cpp'`. This cloud-safe version avoids that problem by using the Python/OpenCV implementation directly.

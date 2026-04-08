# Escáner de Documentos A4 — versión híbrida C++ + Streamlit

Esta versión mantiene la interfaz en **Streamlit/Python** y mueve el núcleo del escaneo a **C++** mediante **pybind11**.

## Qué se ha pasado a C++

- Detección automática del documento.
- Transformación de perspectiva.
- Expansión del cuadrilátero.
- Recorte del marco negro final.
- Escaneo manual a partir de 4 puntos.

## Qué sigue en Python

- Interfaz Streamlit.
- Subida de imágenes.
- Vista previa para clics manuales.
- Descarga del resultado final.

## Estructura

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

## Ejecución local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue en Streamlit Community Cloud

Sube todo el proyecto a GitHub y selecciona `app.py` como archivo principal.

### Importante

Esta versión **necesita compilar un módulo C++** durante el build. Por eso el repositorio incluye:

- `requirements.txt` para dependencias Python.
- `packages.txt` para dependencias del sistema.
- `pyproject.toml` y `CMakeLists.txt` para construir `docscanner_cpp`.

## Nota realista

Este enfoque es técnicamente correcto para usar C++ dentro de Streamlit, pero el punto más delicado del despliegue es la compilación de OpenCV C++ en la nube. Si el build en Streamlit resulta pesado o falla por dependencias del sistema, la alternativa más estable es volver a una versión 100% Python para despliegue y dejar C++ para una app nativa o backend separado.

Built By Alan Masoud

FROM python:3.10-slim

# 1. Instalación de herramientas de compilación
RUN apt-get update && apt-get install -y \
    cmake gcc ninja-build libssl-dev git \
    && rm -rf /var/lib/apt/lists/*

# 2. COMPILACIÓN DE LIBOQS (Rama main para tener los símbolos más recientes)
WORKDIR /liboqs_build
RUN git clone --depth 1 --branch 0.14.0 https://github.com/open-quantum-safe/liboqs.git .
RUN mkdir build && cd build && \
    cmake -GNinja -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
    ninja install

# 3. CONFIGURACIÓN DEL SISTEMA DE LIBRERÍAS
# Registramos las librerías en el sistema para que Python las vea
RUN ldconfig /usr/local/lib

# 4. INSTALACIÓN DE TU ENTORNO CIENTÍFICO CONGELADO
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir liboqs-python==0.14.1

# 5. VARIABLES DE ENTORNO Y DESPLIEGUE
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV PYTHONDONTWRITEBYTECODE=1
COPY main.py .
COPY datia_synth ./datia_synth

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
#!/bin/bash

# Crear el entorno virtual
# Verificar si la carpeta 'venv' existe
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
else
  echo "Virtual environment already exists."
fi

# Activar el entorno virtual
source venv/bin/activate

# Instalar las dependencias
pip install -r src/backend/requirements.txt

# Lanzar el backend
echo "Launching backend..."
cd src/backend
# Asume que el backend se ejecuta con 'python app.py' o algún comando equivalente
python app.py &  # El símbolo '&' ejecuta el backend en segundo plano
BACKEND_PID=$!    # Guardamos el PID del backend para poder detenerlo más tarde

# Volver al directorio principal
cd ../

# Lanzar el frontend
echo "Launching frontend..."
cd frontend

npm ci
npm run dev &  # Ejecuta el frontend en segundo plano
FRONTEND_PID=$!  # Guardamos el PID del frontend para poder detenerlo más tarde

# Función para detener los procesos al finalizar el script
cleanup() {
    echo "Stopping backend..."
    kill $BACKEND_PID
    echo "Stopping frontend..."
    kill $FRONTEND_PID
}

# Atrapamos la señal de interrupción (Ctrl+C) para ejecutar la limpieza
trap cleanup EXIT

# Mantenemos el script corriendo mientras el backend y el frontend están activos
wait $BACKEND_PID
wait $FRONTEND_PID
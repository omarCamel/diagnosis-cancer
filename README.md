# Práctica 8 — CI/CD con GitHub Actions

Proyecto ejecutable para Windows + Visual Studio Code. El modelo incluido fue entrenado con un dataset de **diagnóstico de cáncer de mama**. La práctica está enfocada en el flujo técnico de MLflow, validación y entrega de un modelo, no en decisiones clínicas reales.

## Variables de entorno

Copia el archivo `.env.example` a `.env` y ajusta los valores si necesitas cambiar el experimento o el nombre del modelo a registrar.

```bat
copy .env.example .env
```

Contenido sugerido:

```env
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
EXPERIMENT_NAME=Cancer_Diagnosis
MODEL_NAME=CancerDiagnosisClassifier
```

## 1. Ejecutar localmente (CMD de VS Code)

El modelo incluido requiere scikit-learn 1.8.0 y Python 3.11. Cree el entorno con Python 3.11 para evitar incompatibilidades al cargarlo.

```bat
py -3.11 -m venv .venv311
.venv311\Scripts\activate.bat
python -m pip install -r requirements.txt
python -m pytest -v
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Abrir `http://127.0.0.1:8001/`, `/docs` y `/health`.

> Si el puerto 8000 está ocupado, usa `8001` o cualquier puerto libre. El modo `--reload` puede volver a intentar reusar el mismo puerto y generar conflicto si otra app ya está ejecutándose.

Las predicciones se guardan en `predictions.db` (SQLite) y se pueden consultar en `/predictions?limit=20&offset=0`. El historial requiere la cabecera `X-History-API-Key`; configura la clave antes de iniciar la API:

```powershell
$env:PREDICTIONS_HISTORY_API_KEY = python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Consulta el historial desde PowerShell:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8001/predictions" `
  -Headers @{ "X-History-API-Key" = $env:PREDICTIONS_HISTORY_API_KEY }
```

El historial almacena las 30 características enviadas, además de fecha y resultado. No incluye nombre ni identificador de paciente; aun así, las mediciones son datos sensibles. SQLite está pensado aquí para uso local y no proporciona persistencia adecuada en App Runner. No despliegues esta configuración con tráfico público o datos reales: un despliegue persistente requiere una base gestionada, controles de acceso para las predicciones, y políticas de consentimiento, retención y eliminación.

## 1.1. Entrenar y registrar el modelo con MLflow

El comando de `curl` del wizard es opcional; para arrancar localmente tu servidor de trazas basta con:

```bat
python -m pip install mlflow
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```

Luego, en otra terminal:

```bat
python scripts/train.py
python scripts/register_champion.py
```

Y abrir `http://127.0.0.1:5000` para visualizar experimentos, parámetros y métricas. El experimento principal queda en `Cancer_Diagnosis` y el modelo registrado se llamará `CancerDiagnosisClassifier`.

## 1.2. Orquestar entrenamiento con Apache Airflow

Airflow se ejecuta en Linux; en Windows, esta configuración usa Docker Desktop con contenedores Linux. `Dockerfile.airflow` instala Airflow 2.10.5 para Python 3.11 con el archivo de constraints oficial y las dependencias de entrenamiento de `airflow-requirements.txt`. MLflow corre en una imagen separada definida por `Dockerfile.mlflow` para aislar sus dependencias de las de Airflow. No agregues Airflow a `requirements.txt`, que corresponde a la API local de Windows.

Si este equipo no puede ejecutar Docker/WSL ni habilitar virtualización, la ejecución local de Airflow en Windows no está soportada por el proyecto. Usa un host Linux (local o remoto) con Docker Compose. Desde la raíz del proyecto, inicia Airflow y MLflow:

```powershell
docker compose -f docker-compose.airflow.yml up --build
```

Consulta la salida de inicio para obtener las credenciales generadas por Airflow:

```powershell
docker compose -f docker-compose.airflow.yml logs airflow
```

Abre `http://localhost:8080`, entra al DAG `train_and_register_cancer_model` y selecciona **Trigger DAG**. El DAG ejecuta primero `scripts/train.py` y, si termina correctamente, `scripts/register_champion.py`. MLflow estará disponible en `http://localhost:5000`.

Para detener los servicios, ejecuta en otra terminal:

```powershell
docker compose -f docker-compose.airflow.yml down
```

La ejecución es manual y los datos de Airflow y MLflow se conservan en volúmenes de Docker al detener los servicios.

### Probar la API en un puerto libre

```bat
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Validar:

```bat
curl.exe http://127.0.0.1:8001/health
```

Para probar `POST /predict`, utiliza una fila de `data.csv` sin las columnas `id` y `diagnosis`. Este ejemplo de PowerShell envía la primera fila:

```powershell
$sample = Import-Csv .\data.csv | Select-Object -First 1
$payload = $sample | Select-Object -Property * -ExcludeProperty id, diagnosis
$body = $payload | ConvertTo-Json -Compress
Invoke-RestMethod -Uri "http://127.0.0.1:8001/predict" `
  -Method Post -ContentType "application/json" -Body $body
```

### Estructura recomendada

- `data.csv`: fuente de datos principal
- `scripts/train.py`: entrenamiento del modelo y registro en MLflow
- `models/champion.joblib`: artefacto final del modelo
- `mlruns/`: artefactos producidos por MLflow


## 2. Probar Docker

### Construir y ejecutar la versión 1

```bat
docker build -t diagnosis-cancer:v1 .
docker image ls diagnosis-cancer
docker run -d --name diagnosis-cancer -p 8000:8000 diagnosis-cancer:v1
docker ps
```

Abre `http://localhost:8000/docs` y comprueba la salud de la API:

```powershell
curl.exe http://localhost:8000/health
```

Para probar una predicción con la primera fila del dataset:

```powershell
$sample = Import-Csv .\data.csv | Select-Object -First 1
$payload = $sample | Select-Object -Property * -ExcludeProperty id, diagnosis
$body = $payload | ConvertTo-Json -Compress
Invoke-RestMethod -Uri "http://localhost:8000/predict" `
  -Method Post -ContentType "application/json" -Body $body
```

Consulta los logs y demuestra cómo detener y volver a iniciar el contenedor:

```powershell
docker logs diagnosis-cancer
docker stop diagnosis-cancer
docker ps -a
docker start diagnosis-cancer
curl.exe http://localhost:8000/health
```

### Cambiar el puerto del equipo anfitrión

Elimina el contenedor anterior y publica el puerto interno `8000` en el puerto `9000` del equipo:

```powershell
docker stop diagnosis-cancer
docker rm diagnosis-cancer
docker run -d --name diagnosis-cancer-9000 -p 9000:8000 diagnosis-cancer:v1
curl.exe http://localhost:9000/health
```

La aplicación debe abrir en `http://localhost:9000/docs`; no es necesario cambiar el puerto dentro del contenedor.

### Construir y ejecutar la versión 2

Cambia temporalmente el título en `app/main.py`:

```python
app = FastAPI(title="API de diagnóstico de cáncer - Versión 2", version="1.0", lifespan=lifespan)
```

Construye la nueva imagen, conserva ambas etiquetas y ejecuta la versión 2 en el puerto `9000`:

```powershell
docker build -t diagnosis-cancer:v2 .
docker image ls diagnosis-cancer
docker stop diagnosis-cancer-9000
docker rm diagnosis-cancer-9000
docker run -d --name diagnosis-cancer-v2 -p 9000:8000 diagnosis-cancer:v2
```

Abre `http://localhost:9000/docs` y comprueba que aparece el título de la versión 2. La imagen `diagnosis-cancer:v1` conserva la versión original. Si restauras el título original en el código, no vuelvas a construir `v2`.

### Limpiar los contenedores

Elimina los contenedores de la práctica; las imágenes `v1` y `v2` permanecen disponibles:

```powershell
docker stop diagnosis-cancer-v2
docker rm diagnosis-cancer-v2
```

## 3. Publicar en GitHub

Crear un repositorio vacío llamado `diagnosis-cancer`. Sustituir `TU_USUARIO`:

```bat
git init -b main
git add .
git commit -m "Proyecto inicial CI CD"
git remote add origin https://github.com/TU_USUARIO/diagnosis-cancer.git
git push -u origin main
```

En **Actions**, observar los jobs `test`, `docker` y `deploy`. La publicación ocurre solo en push a `main` después de pasar las pruebas. En Pull Request se ejecutan pruebas, pero no se publican imágenes ni se despliega.

### Despliegue automático en AWS App Runner

El job `docker` publica la imagen en Amazon ECR y GHCR. Después, `deploy` inicia un despliegue del servicio App Runner en cada push a `main` que supera las pruebas. App Runner ejecuta la imagen con acceso público para que la interfaz y la API puedan abrirse desde el navegador.

Configura AWS antes del primer push:

1. Crea un repositorio privado de Amazon ECR llamado `diagnosis-cancer`.
2. Configura el proveedor OIDC de GitHub Actions (`token.actions.githubusercontent.com`) y un rol IAM que confíe solo en `repo:omarCamel/diagnosis-cancer:ref:refs/heads/main`, con audiencia `sts.amazonaws.com`.
3. Concede a ese rol permisos para autenticarse y publicar imágenes en ECR (`ecr:GetAuthorizationToken`, `ecr:BatchCheckLayerAvailability`, `ecr:CompleteLayerUpload`, `ecr:InitiateLayerUpload`, `ecr:PutImage` y `ecr:UploadLayerPart`) y para iniciar despliegues del servicio (`apprunner:StartDeployment`). `ecr:GetAuthorizationToken` requiere `Resource: "*"`, restringe las demás acciones de ECR al repositorio `diagnosis-cancer` y `StartDeployment` al ARN del servicio.
4. En **Settings → Secrets and variables → Actions** del repositorio, añade:
   - Secret `AWS_ROLE_TO_ASSUME`: ARN del rol IAM de GitHub Actions.
   - Variable `AWS_REGION`: región de AWS, por ejemplo `us-east-1`.
5. Ejecuta un push a `main` para publicar la imagen en ECR. Crea el servicio en **AWS App Runner → Create service**, selecciona **Amazon ECR** y la imagen `diagnosis-cancer:latest`, asigna un rol de acceso a ECR con la política administrada `AWSAppRunnerServicePolicyForECRAccess` (la consola puede crearlo), configura el servicio como público y desactiva los despliegues automáticos de ECR para evitar duplicar el job de GitHub. App Runner desplegará esta imagen inicial.
6. Copia el ARN del servicio y guárdalo como variable de Actions `AWS_APP_RUNNER_SERVICE_ARN`. A partir de entonces, cada push a `main` inicia un nuevo despliegue. El ARN no es secreto; si la variable todavía no existe, el job `deploy` se omite.

El workflow usa OpenID Connect y no requiere guardar claves de acceso de AWS en GitHub. La URL pública del servicio aparece en la consola de App Runner. La imagen también sigue publicándose en GHCR.

## 4. Provocar y corregir un fallo

En `tests/test_health.py`, cambiar temporalmente el esperado `200` por `404`. Hacer commit y push. Comprobar que `test` falla y que no se ejecutan `docker` ni `deploy`. Restaurar `200`, hacer otro commit y push; comprobar que los jobs terminan correctamente.

## 5. Descargar desde GHCR

La imagen publicada usa el formato `ghcr.io/tu_usuario/diagnosis-cancer:latest`, en minúsculas. Si el paquete es privado, autenticar Docker con credenciales autorizadas.

```bat
docker pull ghcr.io/tu_usuario/diagnosis-cancer:latest
docker run --rm -p 8000:8000 ghcr.io/tu_usuario/diagnosis-cancer:latest
```

## Notas

- `models/champion.joblib` contiene un pipeline completo de scikit-learn con preprocesamiento y RandomForest/SVM/LogisticRegression.
- El dataset real del proyecto corresponde a diagnóstico de cáncer, con columnas como `diagnosis`, `radius_mean`, `texture_mean` y demás mediciones del tumor.
- El workflow ejecuta pruebas, publica la imagen en GHCR y Amazon ECR, e inicia el despliegue en AWS App Runner en cada push a `main`.
- Para producción, fijar versiones exactas de dependencias, verificar el artefacto y usar datos/modelos validados.

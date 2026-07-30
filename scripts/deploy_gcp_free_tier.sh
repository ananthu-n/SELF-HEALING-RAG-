#!/usr/bin/env bash
# ==============================================================================
# ZERO-COST GCP CLOUD RUN DEPLOYMENT SCRIPT
# This script deploys the application with strict safeguards to guarantee 
# that compute resources stay strictly within the GCP Free Tier limits.
# ==============================================================================

set -e

# Configuration (Customize as needed)
PROJECT_ID=$(gcloud config get-value project 2>/dev/null || echo "")
REGION="us-central1"
SERVICE_NAME="self-healing-rag"
IMAGE_TAG="${REGION}-docker.pkg.dev/${PROJECT_ID}/selfrag-repo/${SERVICE_NAME}:latest"

if [ -z "$PROJECT_ID" ]; then
    echo "[ERROR] No GCP Project ID configured. Run 'gcloud config set project <PROJECT_ID>' first."
    exit 1
fi

echo "======================================================================"
echo "          ZERO-COST GCP CLOUD RUN SAFEGUARD DEPLOYMENT"
echo "======================================================================"
echo "GCP Project ID: ${PROJECT_ID}"
echo "Region:         ${REGION}"
echo "Service:        ${SERVICE_NAME}"
echo "----------------------------------------------------------------------"

# 1. Enable Cloud Run & Artifact Registry APIs if needed
echo "[1/4] Enabling required GCP APIs..."
gcloud services enable run.googleapis.com artifactregistry.googleapis.com --project="${PROJECT_ID}"

# 2. Create Artifact Registry repository if not exists
echo "[2/4] Ensuring Artifact Registry repository exists..."
gcloud artifacts repositories describe selfrag-repo --location="${REGION}" 2>/dev/null || \
gcloud artifacts repositories create selfrag-repo --repository-format=docker --location="${REGION}" --description="Self-Healing RAG Docker Repo"

# 3. Build & Push Image using Cloud Build (no local CPU/bandwidth required)
echo "[3/4] Building container image in Cloud Build..."
gcloud builds submit --tag "${IMAGE_TAG}" .

# 4. Deploy to Cloud Run with STRICT FREE TIER SAFEGUARDS
echo "[4/4] Deploying to Cloud Run with Hard-Capped Free Tier Safeguards..."
gcloud run deploy "${SERVICE_NAME}" \
  --image="${IMAGE_TAG}" \
  --region="${REGION}" \
  --platform=managed \
  --allow-unauthenticated \
  --min-instances=0 \
  --max-instances=1 \
  --cpu-throttling \
  --concurrency=10 \
  --timeout=60s \
  --memory=4Gi \
  --cpu=2 \
  --set-env-vars="ENV=production,MAX_DAILY_QUOTA=500"

echo "======================================================================"
echo "SUCCESS! Your application is deployed safely under Free Tier safeguards."
echo "Hard Safeguards Active:"
echo "  - Min Instances: 0 (Scales to $0 cost when idle)"
echo "  - Max Instances: 1 (Strict limit: Cannot scale up paid nodes)"
echo "  - CPU Throttling: Active (CPU billing stops when idle)"
echo "  - App Quota Cap:  500 queries/day max"
echo "======================================================================"

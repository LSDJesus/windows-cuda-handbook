# Google Cloud SDK & Cloud Development Guide

> **☁️ Setting up Google Cloud SDK for AI/ML development on Windows**
>
> This guide covers Google Cloud SDK installation, authentication, AI Platform setup, and cloud-based ML workflows for Windows developers.

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [SDK Installation](#-sdk-installation)
- [Authentication](#-authentication)
- [AI Platform Setup](#-ai-platform-setup)
- [Cloud Storage](#-cloud-storage)
- [Vertex AI](#-vertex-ai)
- [ML Workflows](#-ml-workflows)
- [Cost Optimization](#-cost-optimization)
- [Troubleshooting](#-troubleshooting)

## 🚀 Quick Start

### Basic Setup
```powershell
# Download and install Google Cloud SDK
# Run installer from https://cloud.google.com/sdk/docs/install

# Initialize SDK
gcloud init

# Authenticate
gcloud auth login

# Set project
gcloud config set project your-project-id
```

### First AI Platform Job
```powershell
# Create a simple training job
gcloud ai-platform jobs submit training my_job \
    --region us-central1 \
    --python-version 3.10 \
    --runtime-version 2.11 \
    --package-path trainer/ \
    --module-name trainer.task \
    --job-dir gs://my-bucket/job-dir \
    -- \
    --train-files gs://my-bucket/data/train.csv \
    --eval-files gs://my-bucket/data/eval.csv
```

### Storage Operations
```powershell
# Create bucket
gsutil mb gs://my-unique-bucket-name

# Upload files
gsutil cp -r local-folder gs://my-bucket/

# Download results
gsutil cp -r gs://my-bucket/results local-results
```

## 🔧 SDK Installation

### Windows Installation

#### Method 1: Interactive Installer
```powershell
# Download from Google Cloud website
# Run GoogleCloudSDKInstaller.exe
# Follow the installation wizard
# Choose installation directory (default: C:\Google\Cloud SDK)
```

#### Method 2: PowerShell Script
```powershell
# Download and run installer script
(New-Object Net.WebClient).DownloadFile("https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe", "$env:TEMP\GoogleCloudSDKInstaller.exe")
& "$env:TEMP\GoogleCloudSDKInstaller.exe"
```

#### Method 3: Chocolatey
```powershell
# Install via Chocolatey
choco install gcloudsdk

# Refresh environment
refreshenv
```

### Post-Installation Setup

#### Environment Variables
```powershell
# Add to PATH (usually done automatically)
$env:PATH = "C:\Google\Cloud SDK\google-cloud-sdk\bin;$env:PATH"

# Set CLOUDSDK_PYTHON
$env:CLOUDSDK_PYTHON = "C:\Python310\python.exe"
```

#### Verify Installation
```powershell
# Check version
gcloud version

# List components
gcloud components list

# Update SDK
gcloud components update
```

### Component Installation

#### Essential Components
```powershell
# Install AI Platform component
gcloud components install ai-platform

# Install Cloud Storage tools
gcloud components install gsutil

# Install Kubernetes tools
gcloud components install kubectl

# Install App Engine tools
gcloud components install app-engine-python
```

#### Beta Components
```powershell
# Install Vertex AI (beta)
gcloud components install beta

# Install Cloud Build
gcloud components install cloud-build-local
```

## 🔐 Authentication

### Authentication Methods

#### Interactive Login
```powershell
# Browser-based authentication
gcloud auth login

# List accounts
gcloud auth list

# Set active account
gcloud auth login your-email@gmail.com
```

#### Service Account Authentication
```powershell
# Create service account key
gcloud iam service-accounts keys create key.json \
    --iam-account my-service-account@my-project.iam.gserviceaccount.com

# Activate service account
gcloud auth activate-service-account --key-file key.json

# Set as default
gcloud config set account my-service-account@my-project.iam.gserviceaccount.com
```

#### Application Default Credentials
```powershell
# For local development
gcloud auth application-default login

# For service accounts
gcloud auth application-default login --client-id-file key.json
```

### Project Configuration

#### Project Setup
```powershell
# List available projects
gcloud projects list

# Set default project
gcloud config set project my-project-id

# Verify configuration
gcloud config list
```

#### Multiple Projects
```powershell
# Create configuration for different projects
gcloud config configurations create dev
gcloud config configurations create prod

# Switch configurations
gcloud config configurations activate dev

# Set project for each config
gcloud config set project dev-project
gcloud config set project prod-project
```

## 🤖 AI Platform Setup

### Legacy AI Platform

#### Training Job Setup
```powershell
# Submit training job
gcloud ai-platform jobs submit training my_training_job \
    --region us-central1 \
    --python-version 3.10 \
    --runtime-version 2.11 \
    --package-path ./trainer \
    --module-name trainer.task \
    --job-dir gs://my-bucket/job-dir \
    --config config.yaml \
    -- \
    --train-data gs://my-bucket/data/train.csv \
    --eval-data gs://my-bucket/data/eval.csv \
    --model-dir gs://my-bucket/model
```

#### Job Configuration
```yaml
# config.yaml
trainingInput:
  pythonVersion: '3.10'
  runtimeVersion: '2.11'
  jobDir: gs://my-bucket/job-dir
  pythonModule: trainer.task
  packageUris:
    - gs://my-bucket/trainer-0.1.tar.gz
  region: us-central1
  args:
    - --train-data
    - gs://my-bucket/data/train.csv
  hyperparameters:
    goal: MAXIMIZE
    hyperparameterMetricTag: accuracy
    maxTrials: 10
    maxParallelTrials: 2
    params:
      - parameterName: learning-rate
        type: DOUBLE
        minValue: 0.001
        maxValue: 0.1
        scaleType: UNIT_LOG_SCALE
```

#### Custom Container
```powershell
# Build custom container
gcloud builds submit --tag gcr.io/my-project/my-trainer .

# Submit job with custom container
gcloud ai-platform jobs submit training my_job \
    --region us-central1 \
    --master-image-uri gcr.io/my-project/my-trainer \
    --job-dir gs://my-bucket/job-dir
```

### Vertex AI Migration

#### Vertex AI Training
```powershell
# Submit custom training job
gcloud ai custom-jobs create \
    --region us-central1 \
    --display-name my-training-job \
    --python-package-uris gs://my-bucket/trainer-0.1.tar.gz \
    --python-module trainer.task \
    --args --train-data=gs://my-bucket/data/train.csv,--model-dir=gs://my-bucket/model
```

#### Managed Dataset
```powershell
# Create dataset
gcloud ai datasets create my-dataset \
    --display-name "My Dataset" \
    --metadata-schema-uri gs://google-cloud-aiplatform/schema/dataset/metadata/image_1.0.0.yaml \
    --region us-central1

# Import data
gcloud ai datasets import-data my-dataset \
    --metadata-schema-uri gs://google-cloud-aiplatform/schema/dataset/metadata/image_1.0.0.yaml \
    --input-configs '[{"gcsSource": {"uris": ["gs://my-bucket/data.csv"]}}]'
```

## ☁️ Cloud Storage

### Bucket Management

#### Creating Buckets
```powershell
# Create bucket
gsutil mb -p my-project -c regional -l us-central1 gs://my-bucket

# Create bucket with versioning
gsutil mb -p my-project gs://my-versioned-bucket
gsutil versioning set on gs://my-versioned-bucket
```

#### Bucket Permissions
```powershell
# Make bucket public
gsutil iam ch allUsers:objectViewer gs://my-public-bucket

# Set lifecycle policy
gsutil lifecycle set lifecycle.json gs://my-bucket

# lifecycle.json
{
  "rule": [
    {
      "action": {"type": "Delete"},
      "condition": {
        "age": 365,
        "isLive": true
      }
    }
  ]
}
```

### Data Transfer

#### Upload Operations
```powershell
# Upload single file
gsutil cp local-file.txt gs://my-bucket/

# Upload directory
gsutil cp -r local-folder gs://my-bucket/

# Upload with compression
gsutil cp -Z large-file.dat gs://my-bucket/

# Parallel upload
gsutil -m cp -r local-folder gs://my-bucket/
```

#### Download Operations
```powershell
# Download single file
gsutil cp gs://my-bucket/file.txt local-file.txt

# Download directory
gsutil cp -r gs://my-bucket/folder local-folder

# Download with resume
gsutil cp -r -L resume-file gs://my-bucket/large-folder local-folder
```

#### Sync Operations
```powershell
# Sync local to cloud
gsutil rsync -r local-folder gs://my-bucket/folder

# Sync cloud to local
gsutil rsync -r gs://my-bucket/folder local-folder

# Mirror sync
gsutil rsync -r -d local-folder gs://my-bucket/folder
```

### Storage Optimization

#### Storage Classes
```powershell
# Set storage class
gsutil rewrite -s nearline gs://my-bucket/file.txt
gsutil rewrite -s coldline gs://my-bucket/archive.txt
gsutil rewrite -s archive gs://my-bucket/backup.txt
```

#### Cost Monitoring
```powershell
# Check bucket size
gsutil du -sh gs://my-bucket

# List large files
gsutil ls -lh gs://my-bucket | sort -k2 -h | tail -10

# Check storage costs
gcloud billing accounts list
gcloud billing projects link my-project --billing-account=XXXXXX-XXXXXX-XXXXXX
```

## 🎯 Vertex AI

### Model Training

#### AutoML Training
```powershell
# Create dataset
gcloud ai datasets create my-image-dataset \
    --display-name "Image Dataset" \
    --metadata-schema-uri gs://google-cloud-aiplatform/schema/dataset/metadata/image_1.0.0.yaml

# Import data
gcloud ai datasets import-data my-image-dataset \
    --metadata-schema-uri gs://google-cloud-aiplatform/schema/dataset/metadata/image_1.0.0.yaml \
    --input-configs '[{"gcsSource": {"uris": ["gs://my-bucket/images/"]}}]'

# Create training job
gcloud ai custom-jobs create \
    --region us-central1 \
    --display-name my-training-job \
    --python-package-uris gs://my-bucket/trainer.tar.gz \
    --python-module trainer.task \
    --machine-type n1-standard-4 \
    --replica-count 1
```

#### Custom Container Training
```powershell
# Build container
gcloud builds submit --tag gcr.io/my-project/my-trainer .

# Create training job
gcloud ai custom-jobs create \
    --region us-central1 \
    --display-name container-training \
    --worker-pool-spec machine-type=n1-standard-4,replica-count=1,container-image-uri=gcr.io/my-project/my-trainer
```

### Model Deployment

#### Endpoint Creation
```powershell
# Create endpoint
gcloud ai endpoints create --display-name my-endpoint --region us-central1

# Deploy model
gcloud ai endpoints deploy-model my-endpoint \
    --model my-model \
    --display-name my-deployment \
    --machine-type n1-standard-2 \
    --min-replica-count 1 \
    --max-replica-count 3 \
    --traffic-split 0=100
```

#### Online Prediction
```powershell
# Make prediction
gcloud ai endpoints predict my-endpoint \
    --region us-central1 \
    --json-request request.json

# request.json
{
  "instances": [
    {
      "input": [1, 2, 3, 4]
    }
  ]
}
```

### Batch Prediction
```powershell
# Create batch prediction job
gcloud ai batch-prediction-jobs create \
    --region us-central1 \
    --display-name my-batch-job \
    --input-uris gs://my-bucket/input/* \
    --output-uri gs://my-bucket/output/ \
    --model my-model \
    --machine-type n1-standard-2 \
    --starting-replica-count 1
```

## 🔄 ML Workflows

### Cloud Build Integration

#### Build Configuration
```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/my-app', '.']

  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/my-app']

  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: 'gcloud'
    args:
      - 'ai'
      - 'custom-jobs'
      - 'create'
      - '--region=us-central1'
      - '--display-name=my-training-job'
      - '--python-package-uris=gs://my-bucket/trainer.tar.gz'
      - '--python-module=trainer.task'
```

#### Trigger Build
```powershell
# Submit build
gcloud builds submit --config cloudbuild.yaml .

# Create build trigger
gcloud builds triggers create github \
    --repo-name my-repo \
    --repo-owner my-org \
    --branch-pattern main \
    --build-config cloudbuild.yaml
```

### CI/CD Pipeline

#### GitHub Actions Integration
```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline
on:
  push:
    branches: [main]

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v1
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v1

      - name: Submit training job
        run: |
          gcloud ai custom-jobs create \
            --region us-central1 \
            --display-name ci-training-${{ github.run_number }} \
            --python-package-uris gs://my-bucket/trainer.tar.gz \
            --python-module trainer.task
```

### Monitoring and Logging

#### Job Monitoring
```powershell
# List training jobs
gcloud ai custom-jobs list --region us-central1

# Get job details
gcloud ai custom-jobs describe my-job --region us-central1

# Stream logs
gcloud ai custom-jobs stream-logs my-job --region us-central1
```

#### Cloud Logging
```powershell
# View logs
gcloud logging read "resource.type=ml_job" --limit 10

# Filter by job
gcloud logging read "resource.labels.job_id=my-job" --limit 10

# Export logs
gcloud logging read "resource.type=ml_job" --format export > logs.json
```

## 💰 Cost Optimization

### Resource Management

#### Instance Selection
```powershell
# List available machine types
gcloud compute machine-types list --zones us-central1-a

# Use spot instances for training
gcloud ai custom-jobs create \
    --region us-central1 \
    --worker-pool-spec machine-type=n1-standard-4,replica-count=1,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1 \
    --scheduling spot
```

#### Auto-scaling
```powershell
# Set up auto-scaling for endpoints
gcloud ai endpoints deploy-model my-endpoint \
    --model my-model \
    --min-replica-count 1 \
    --max-replica-count 10 \
    --autoscaling-metric-cpu-utilization-target 70
```

### Cost Monitoring

#### Billing Alerts
```powershell
# Set up budget
gcloud billing budgets create my-budget \
    --billing-account XXXX-XXXX-XXXX \
    --display-name "ML Budget" \
    --amount 1000 \
    --threshold-rule percent=50,percent=90,percent=100

# Create budget notification
gcloud billing budgets create my-budget-notification \
    --billing-account XXXX-XXXX-XXXX \
    --budget my-budget \
    --threshold-rule percent=50 \
    --pubsub-topic projects/my-project/topics/budget-alerts
```

#### Cost Analysis
```powershell
# View costs by service
gcloud billing export create my-export \
    --billing-account XXXX-XXXX-XXXX \
    --export-bucket gs://my-billing-export \
    --dataset my-dataset

# Query costs
bq query --use_legacy_sql=false \
    'SELECT service.description, SUM(cost) as total_cost
     FROM `my-project.my-dataset.gcp_billing_export_*`
     WHERE _PARTITIONTIME >= "2024-01-01"
     GROUP BY service.description
     ORDER BY total_cost DESC'
```

## 🔧 Troubleshooting

### Common Issues

#### Issue 1: Authentication Problems
**Problem:** `gcloud auth login` fails

**Solutions:**
```powershell
# Clear authentication
gcloud auth revoke

# Try different browser
gcloud auth login --no-launch-browser

# Check proxy settings
gcloud config set proxy/type http
gcloud config set proxy/address proxy.company.com
gcloud config set proxy/port 8080
```

#### Issue 2: Permission Errors
**Problem:** Access denied to resources

**Solutions:**
```powershell
# Check current account
gcloud auth list

# Verify project access
gcloud projects get-iam-policy my-project

# Add required roles
gcloud projects add-iam-policy-binding my-project \
    --member user:my-email@gmail.com \
    --role roles/ml.admin
```

#### Issue 3: Job Failures
**Problem:** Training job fails

**Solutions:**
```powershell
# Check job logs
gcloud ai custom-jobs stream-logs my-job --region us-central1

# Get detailed error
gcloud ai custom-jobs describe my-job --region us-central1

# Check quota limits
gcloud compute regions describe us-central1 --format "value(quotas[0].metric,quotas[0].limit,quotas[0].usage)"
```

#### Issue 4: Network Issues
**Problem:** Slow uploads/downloads

**Solutions:**
```powershell
# Use parallel transfers
gsutil -m cp -r local-folder gs://my-bucket/

# Adjust chunk size
gsutil -o GSUtil:parallel_composite_upload_threshold=150M cp large-file gs://my-bucket/

# Check network speed
gcloud compute instances create test-instance --zone us-central1-a --machine-type f1-micro
```

### Debug Tools

#### SDK Debug Mode
```powershell
# Enable debug logging
gcloud config set verbosity debug

# Run command with debug
gcloud ai custom-jobs list --log-http

# View debug logs
gcloud logging read "resource.type=global" --filter "severity>=DEBUG"
```

#### Network Diagnostics
```powershell
# Test connectivity
gcloud auth login --verbosity debug

# Check DNS resolution
nslookup storage.googleapis.com

# Test API endpoints
curl -v https://ml.googleapis.com/v1/projects/my-project/locations/us-central1/trainingPipelines
```

---

*Google Cloud SDK provides powerful tools for AI/ML development. This guide covers the essential setup and workflows for Windows developers working with cloud-based ML projects.*
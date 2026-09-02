# 🏗️ Terraform Deployment — text2sql-toolkit

Production-grade Infrastructure-as-Code templates for deploying **text2sql-toolkit** on the three major cloud providers.

| Cloud | Service | Profile Cache | Secrets | Directory |
|-------|---------|---------------|---------|-----------|
| [AWS](#-aws-lambda--api-gateway) | Lambda + API Gateway v2 | S3 | Environment vars | [`terraform/aws/`](./aws/) |
| [GCP](#-gcp-cloud-run) | Cloud Run | GCS | Secret Manager | [`terraform/gcp/`](./gcp/) |
| [Azure](#-azure-container-apps) | Container Apps | Blob Storage | Key Vault | [`terraform/azure/`](./azure/) |

---

## 📋 Prerequisites

All three templates require:

1. **Terraform ≥ 1.5** — [Install guide](https://developer.hashicorp.com/terraform/install)
2. **Docker** — For building and pushing container images
3. **Cloud CLI** — `aws`, `gcloud`, or `az` depending on your target
4. **LLM API Key** — e.g. OpenAI, Anthropic, Google

```bash
# Verify Terraform is installed
terraform --version   # Must be >= 1.5

# Verify Docker is running
docker info
```

---

## ☁️ AWS — Lambda + API Gateway

### Architecture

```
Internet → API Gateway v2 (HTTP API)
               ↓
           AWS Lambda (container image from ECR)
               ↓                ↓
         Target Database    S3 (profile cache)
```

**Resources created:**
- ECR repository (container registry)
- Lambda function (container-based, up to 10 GB image)
- Lambda Function URL with response streaming
- API Gateway v2 (HTTP API) with auto-deploy
- S3 bucket (profile cache, versioned, encrypted)
- IAM role + policies (least-privilege)
- CloudWatch log groups (Lambda + API Gateway)
- Optional: VPC security group for private database access

### Step-by-Step Deployment

#### 1. Authenticate with AWS

```bash
# Option A: AWS CLI profile
aws configure --profile text2sql
export AWS_PROFILE=text2sql

# Option B: Environment variables
export AWS_ACCESS_KEY_ID="AKIA..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"
```

#### 2. Build and Push the Docker Image

```bash
# Navigate to the project root (not the terraform directory)
cd /path/to/text2sql-toolkit

# Initialize Terraform first to get the ECR repository URL
cd terraform/aws
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values

terraform init
terraform apply -target=aws_ecr_repository.text2sql

# Get the ECR repository URL from the output
ECR_URL=$(terraform output -raw ecr_repository_url)

# Authenticate Docker with ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin $ECR_URL

# Build and push the Lambda image
cd ../..
docker build -f Dockerfile.lambda -t $ECR_URL:latest .
docker push $ECR_URL:latest
```

#### 3. Configure Variables

```bash
cd terraform/aws
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars`:

```hcl
aws_region     = "us-east-1"
environment    = "dev"
openai_api_key = "sk-your-real-key"
llm_model      = "gpt-4o-mini"
db_uri         = "postgresql://user:pass@your-rds-host:5432/mydb"

# For private RDS access:
enable_vpc = true
vpc_id     = "vpc-0123456789abcdef0"
subnet_ids = ["subnet-aaa", "subnet-bbb"]
```

#### 4. Deploy

```bash
terraform init
terraform plan     # Review the plan carefully
terraform apply    # Type 'yes' to confirm
```

#### 5. Test

```bash
# Get the endpoints
API_URL=$(terraform output -raw api_gateway_url)
LAMBDA_URL=$(terraform output -raw lambda_function_url)

# Test synchronous endpoint
curl -X POST "$API_URL/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "How many users?"}'

# Test SSE streaming endpoint
curl -X POST "$API_URL/ask/stream" \
  -H "Content-Type: application/json" \
  -d '{"question": "How many users?"}'

# Direct Lambda URL (supports response streaming)
curl -X POST "$LAMBDA_URL/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "How many users?"}'
```

#### 6. Update (Redeploy)

```bash
# Rebuild and push new image
cd /path/to/text2sql-toolkit
docker build -f Dockerfile.lambda -t $ECR_URL:latest .
docker push $ECR_URL:latest

# Force Lambda to pull the new image
aws lambda update-function-code \
  --function-name text2sql-dev \
  --image-uri $ECR_URL:latest
```

#### 7. Tear Down

```bash
cd terraform/aws
terraform destroy   # Type 'yes' to confirm
```

### Cost Estimate (AWS)

| Resource | Pricing | Estimate (1K queries/month) |
|----------|---------|---------------------------|
| Lambda | $0.20/1M requests + compute | ~$2–5 |
| API Gateway | $1.00/1M requests | ~$0.001 |
| S3 | $0.023/GB/month | ~$0.01 |
| CloudWatch | $0.50/GB ingested | ~$0.50 |
| **Total** | | **~$3–6/month** |

---

## 🌐 GCP — Cloud Run

### Architecture

```
Internet → Cloud Run (container from Artifact Registry)
               ↓                ↓              ↓
         Target Database    GCS (cache)    Secret Manager
```

**Resources created:**
- Artifact Registry repository
- Cloud Run v2 service (autoscaling, health probes)
- GCS bucket (profile cache, versioned)
- Secret Manager secret (API key)
- Service account (least-privilege IAM)
- Required API enablement (automated)

### Step-by-Step Deployment

#### 1. Authenticate with GCP

```bash
# Login and set project
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

#### 2. Build and Push the Docker Image

```bash
# Navigate to the project root
cd /path/to/text2sql-toolkit

# Initialize Terraform to get the Artifact Registry URL
cd terraform/gcp
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values

terraform init
terraform apply -target=google_artifact_registry_repository.text2sql \
                -target=google_project_service.apis

# Get the Artifact Registry URL
AR_URL=$(terraform output -raw artifact_registry_url)

# Configure Docker for Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev

# Build and push (using the standard Dockerfile, not Lambda)
cd ../..
docker build -t $AR_URL/text2sql:latest .
docker push $AR_URL/text2sql:latest
```

> **Note:** Cloud Run uses the standard `Dockerfile` (FastAPI server), not `Dockerfile.lambda`.

#### 3. Configure Variables

```bash
cd terraform/gcp
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars`:

```hcl
gcp_project_id = "my-project-123"
gcp_region     = "us-central1"
environment    = "dev"
openai_api_key = "sk-your-real-key"
llm_model      = "gpt-4o-mini"
db_uri         = "postgresql://user:pass@/mydb?host=/cloudsql/project:region:instance"

# Cloud Run scaling
cloud_run_min_instances = 0    # Scale to zero
cloud_run_max_instances = 10
allow_unauthenticated   = true # Set false for prod
```

#### 4. Deploy

```bash
terraform init
terraform plan
terraform apply
```

#### 5. Test

```bash
# Get the Cloud Run URL
SERVICE_URL=$(terraform output -raw cloud_run_url)

# Test synchronous endpoint
curl -X POST "$SERVICE_URL/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "How many users?"}'

# Test SSE streaming endpoint
curl -N -X POST "$SERVICE_URL/ask/stream" \
  -H "Content-Type: application/json" \
  -d '{"question": "How many users?", "event_verbosity": "verbose"}'
```

#### 6. Update (Redeploy)

```bash
# Rebuild and push
cd /path/to/text2sql-toolkit
docker build -t $AR_URL/text2sql:latest .
docker push $AR_URL/text2sql:latest

# Cloud Run auto-deploys from :latest on next revision
# Or force a new revision:
gcloud run services update text2sql-dev \
  --image=$AR_URL/text2sql:latest \
  --region=us-central1
```

#### 7. Tear Down

```bash
cd terraform/gcp
terraform destroy
```

### Cost Estimate (GCP)

| Resource | Pricing | Estimate (1K queries/month) |
|----------|---------|---------------------------|
| Cloud Run | $0.00002400/vCPU-sec + $0.00000250/GiB-sec | ~$1–3 |
| Artifact Registry | $0.10/GB/month | ~$0.05 |
| GCS | $0.020/GB/month | ~$0.01 |
| Secret Manager | $0.06/10K access | ~$0.006 |
| **Total** | | **~$1–4/month** |

---

## 🔵 Azure — Container Apps

### Architecture

```
Internet → Container Apps (image from ACR)
               ↓              ↓             ↓
         Target Database   Blob Storage   Key Vault
```

**Resources created:**
- Resource Group
- Azure Container Registry (ACR)
- Container Apps Environment + Container App (autoscaling)
- Storage Account + Blob container (profile cache)
- Key Vault (secrets)
- User-assigned Managed Identity (least-privilege)
- Log Analytics workspace

### Step-by-Step Deployment

#### 1. Authenticate with Azure

```bash
# Login
az login

# Set subscription (if you have multiple)
az account set --subscription "YOUR_SUBSCRIPTION_ID"
```

#### 2. Build and Push the Docker Image

```bash
# Navigate to the project root
cd /path/to/text2sql-toolkit

# Initialize Terraform to create ACR
cd terraform/azure
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values

terraform init
terraform apply -target=azurerm_resource_group.text2sql \
                -target=azurerm_container_registry.text2sql

# Get the ACR login server
ACR_SERVER=$(terraform output -raw acr_login_server)

# Login to ACR
az acr login --name $(echo $ACR_SERVER | cut -d. -f1)

# Build and push
cd ../..
docker build -t $ACR_SERVER/text2sql:latest .
docker push $ACR_SERVER/text2sql:latest
```

#### 3. Configure Variables

```bash
cd terraform/azure
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars`:

```hcl
azure_location = "East US"
environment    = "dev"
openai_api_key = "sk-your-real-key"
llm_model      = "gpt-4o-mini"
db_uri         = "postgresql://user:pass@your-db.postgres.database.azure.com:5432/mydb"

# Container Apps scaling
min_replicas     = 0     # Scale to zero
max_replicas     = 10
container_cpu    = 1
container_memory = "2Gi"
```

#### 4. Deploy

```bash
terraform init
terraform plan
terraform apply
```

#### 5. Test

```bash
# Get the Container App URL
APP_URL=$(terraform output -raw container_app_url)

# Test synchronous endpoint
curl -X POST "$APP_URL/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "How many users?"}'

# Test SSE streaming endpoint
curl -N -X POST "$APP_URL/ask/stream" \
  -H "Content-Type: application/json" \
  -d '{"question": "How many users?", "event_verbosity": "verbose"}'
```

#### 6. Update (Redeploy)

```bash
# Rebuild and push
cd /path/to/text2sql-toolkit
docker build -t $ACR_SERVER/text2sql:latest .
docker push $ACR_SERVER/text2sql:latest

# Create new revision
az containerapp update \
  --name text2sql-dev \
  --resource-group text2sql-dev-rg \
  --image $ACR_SERVER/text2sql:latest
```

#### 7. Tear Down

```bash
cd terraform/azure
terraform destroy
```

### Cost Estimate (Azure)

| Resource | Pricing | Estimate (1K queries/month) |
|----------|---------|---------------------------|
| Container Apps | $0.000012/vCPU-sec + $0.000002/GiB-sec | ~$1–3 |
| ACR (Basic) | $5/month | $5 |
| Blob Storage | $0.018/GB/month | ~$0.01 |
| Key Vault | $0.03/10K operations | ~$0.003 |
| Log Analytics | $2.76/GB ingested | ~$0.50 |
| **Total** | | **~$7–9/month** |

---

## 🔒 Security Best Practices

### Secrets Management

| Practice | AWS | GCP | Azure |
|----------|-----|-----|-------|
| **API keys** | Lambda env vars (encrypted at rest) | Secret Manager | Key Vault |
| **DB credentials** | Secrets Manager + VPC | Secret Manager + Cloud SQL IAM | Key Vault + Private Link |
| **Container images** | ECR (scan on push) | Artifact Registry | ACR |

### Network Isolation

- **AWS:** Deploy Lambda in VPC with private subnets for RDS access. Use VPC endpoints for S3.
- **GCP:** Use Cloud SQL Auth Proxy or VPC Connector for private DB access.
- **Azure:** Use Private Endpoints for database and Key Vault access. Enable VNet integration on Container Apps.

### Least-Privilege IAM

All templates create **dedicated service identities** with minimal permissions:

```
✅ Read/write profile cache bucket
✅ Read secrets (API keys)
✅ Pull container images
✅ Write logs
❌ No admin access
❌ No cross-service access
```

---

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
name: Deploy text2sql
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: "1.7"

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1

      - name: Login to ECR
        id: ecr
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build and push
        run: |
          ECR_URL=${{ steps.ecr.outputs.registry }}/text2sql-dev-lambda
          docker build -f Dockerfile.lambda -t $ECR_URL:${{ github.sha }} .
          docker push $ECR_URL:${{ github.sha }}

      - name: Terraform Apply
        run: |
          cd terraform/aws
          terraform init
          terraform apply -auto-approve \
            -var="openai_api_key=${{ secrets.OPENAI_API_KEY }}"

      - name: Update Lambda
        run: |
          aws lambda update-function-code \
            --function-name text2sql-dev \
            --image-uri $ECR_URL:${{ github.sha }}
```

---

## 📁 Directory Structure

```
terraform/
├── README.md                        # This file
├── aws/
│   ├── main.tf                      # Lambda + API Gateway + ECR + S3
│   └── terraform.tfvars.example     # Variable template
├── gcp/
│   ├── main.tf                      # Cloud Run + Artifact Registry + GCS
│   └── terraform.tfvars.example     # Variable template
└── azure/
    ├── main.tf                      # Container Apps + ACR + Blob + Key Vault
    └── terraform.tfvars.example     # Variable template
```

---

## 🆘 Troubleshooting

### Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Error: No valid credential sources found` | AWS/GCP/Azure CLI not authenticated | Run `aws configure` / `gcloud auth login` / `az login` |
| `Error: image not found` | Container image not pushed to registry | Follow Step 2 (Build & Push) before deploying |
| Lambda timeout | Query too complex or DB unreachable | Increase `lambda_timeout_seconds` or check VPC/security group |
| Cloud Run cold start slow | Scale-to-zero with heavy image | Set `cloud_run_min_instances = 1` |
| `403 Forbidden` on Cloud Run | `allow_unauthenticated = false` | Set `true` for testing, use IAM auth for prod |
| Container App crashing | Secrets not accessible | Check Managed Identity has Key Vault access |

### Viewing Logs

```bash
# AWS
aws logs tail /aws/lambda/text2sql-dev --follow

# GCP
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=text2sql-dev" --limit=50

# Azure
az containerapp logs show --name text2sql-dev --resource-group text2sql-dev-rg --follow
```

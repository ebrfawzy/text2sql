# ── text2sql-toolkit — GCP Cloud Run Deployment ─────────────────────
#
# Deploys:
#   • Artifact Registry repository for container images
#   • Cloud Run service with autoscaling
#   • GCS bucket for profile cache
#   • IAM service account with least-privilege
#   • Secret Manager for API keys
#
# Usage:
#   terraform init
#   terraform plan -var="openai_api_key=sk-..."
#   terraform apply
# ─────────────────────────────────────────────────────────────────────

terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  # Uncomment for remote state (recommended for teams)
  # backend "gcs" {
  #   bucket = "your-terraform-state-bucket"
  #   prefix = "text2sql/terraform"
  # }
}

provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}

# ── Variables ────────────────────────────────────────────────────────

variable "gcp_project_id" {
  description = "GCP project ID"
  type        = string
}

variable "gcp_region" {
  description = "GCP region for deployment"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name used in resource naming"
  type        = string
  default     = "text2sql"
}

variable "openai_api_key" {
  description = "OpenAI API key (or other LLM provider key)"
  type        = string
  sensitive   = true
}

variable "llm_model" {
  description = "LiteLLM model string"
  type        = string
  default     = "gpt-4o-mini"
}

variable "cloud_run_cpu" {
  description = "CPU allocation for Cloud Run (e.g. '1', '2', '4')"
  type        = string
  default     = "2"
}

variable "cloud_run_memory" {
  description = "Memory allocation for Cloud Run (e.g. '1Gi', '2Gi')"
  type        = string
  default     = "1Gi"
}

variable "cloud_run_min_instances" {
  description = "Minimum number of Cloud Run instances (0 = scale to zero)"
  type        = number
  default     = 0
}

variable "cloud_run_max_instances" {
  description = "Maximum number of Cloud Run instances"
  type        = number
  default     = 10
}

variable "cloud_run_timeout_seconds" {
  description = "Request timeout in seconds (max 3600)"
  type        = number
  default     = 900
}

variable "db_uri" {
  description = "Database URI for the target database"
  type        = string
  default     = "sqlite:///example.db"
  sensitive   = true
}

variable "allow_unauthenticated" {
  description = "Allow unauthenticated access to Cloud Run (disable for prod)"
  type        = bool
  default     = true
}

locals {
  prefix = "${var.project_name}-${var.environment}"
}

# ── Enable Required APIs ────────────────────────────────────────────

resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "secretmanager.googleapis.com",
    "cloudbuild.googleapis.com",
  ])

  project            = var.gcp_project_id
  service            = each.value
  disable_on_destroy = false
}

# ── Artifact Registry ──────────────────────────────────────────────

resource "google_artifact_registry_repository" "text2sql" {
  repository_id = "${local.prefix}-docker"
  location      = var.gcp_region
  format        = "DOCKER"
  description   = "Container images for text2sql-toolkit"

  cleanup_policies {
    id     = "keep-recent"
    action = "KEEP"
    most_recent_versions {
      keep_count = 5
    }
  }

  depends_on = [google_project_service.apis["artifactregistry.googleapis.com"]]
}

# ── GCS Profile Cache ──────────────────────────────────────────────

resource "google_storage_bucket" "profile_cache" {
  name                        = "${local.prefix}-profile-cache-${var.gcp_project_id}"
  location                    = var.gcp_region
  force_destroy               = var.environment != "prod"
  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }
}

# ── Secret Manager (API Key) ──────────────────────────────────────

resource "google_secret_manager_secret" "openai_key" {
  secret_id = "${local.prefix}-openai-key"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis["secretmanager.googleapis.com"]]
}

resource "google_secret_manager_secret_version" "openai_key" {
  secret      = google_secret_manager_secret.openai_key.id
  secret_data = var.openai_api_key
}

# ── Service Account ────────────────────────────────────────────────

resource "google_service_account" "text2sql" {
  account_id   = "${local.prefix}-sa"
  display_name = "text2sql Cloud Run Service Account"
}

resource "google_storage_bucket_iam_member" "cache_access" {
  bucket = google_storage_bucket.profile_cache.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.text2sql.email}"
}

resource "google_secret_manager_secret_iam_member" "openai_key_access" {
  secret_id = google_secret_manager_secret.openai_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.text2sql.email}"
}

# ── Cloud Run Service ──────────────────────────────────────────────

resource "google_cloud_run_v2_service" "text2sql" {
  name     = local.prefix
  location = var.gcp_region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.text2sql.email

    scaling {
      min_instance_count = var.cloud_run_min_instances
      max_instance_count = var.cloud_run_max_instances
    }

    timeout = "${var.cloud_run_timeout_seconds}s"

    containers {
      image = "${var.gcp_region}-docker.pkg.dev/${var.gcp_project_id}/${google_artifact_registry_repository.text2sql.repository_id}/${var.project_name}:latest"

      ports {
        container_port = 8000
      }

      resources {
        limits = {
          cpu    = var.cloud_run_cpu
          memory = var.cloud_run_memory
        }
        startup_cpu_boost = true
      }

      env {
        name  = "TEXT2SQL_MODEL"
        value = var.llm_model
      }

      env {
        name  = "TEXT2SQL_DB_URI"
        value = var.db_uri
      }

      env {
        name  = "TEXT2SQL_PROFILE_CACHE_DIR"
        value = "gs://${google_storage_bucket.profile_cache.name}/profiles"
      }

      env {
        name  = "TEXT2SQL_LOG_LEVEL"
        value = var.environment == "prod" ? "WARNING" : "INFO"
      }

      env {
        name = "OPENAI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.openai_key.secret_id
            version = "latest"
          }
        }
      }

      startup_probe {
        http_get {
          path = "/health"
          port = 8000
        }
        initial_delay_seconds = 5
        period_seconds        = 10
        failure_threshold     = 3
      }

      liveness_probe {
        http_get {
          path = "/health"
          port = 8000
        }
        period_seconds    = 30
        failure_threshold = 3
      }
    }
  }

  depends_on = [
    google_project_service.apis["run.googleapis.com"],
    google_secret_manager_secret_version.openai_key,
  ]
}

# ── Public Access (optional) ────────────────────────────────────────

resource "google_cloud_run_v2_service_iam_member" "public" {
  count    = var.allow_unauthenticated ? 1 : 0
  project  = var.gcp_project_id
  location = google_cloud_run_v2_service.text2sql.location
  name     = google_cloud_run_v2_service.text2sql.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ── Outputs ─────────────────────────────────────────────────────────

output "artifact_registry_url" {
  description = "Artifact Registry URL for pushing Docker images"
  value       = "${var.gcp_region}-docker.pkg.dev/${var.gcp_project_id}/${google_artifact_registry_repository.text2sql.repository_id}"
}

output "cloud_run_url" {
  description = "Cloud Run service URL"
  value       = google_cloud_run_v2_service.text2sql.uri
}

output "gcs_profile_cache" {
  description = "GCS bucket for profile cache"
  value       = "gs://${google_storage_bucket.profile_cache.name}/profiles"
}

output "service_account_email" {
  description = "Service account email"
  value       = google_service_account.text2sql.email
}

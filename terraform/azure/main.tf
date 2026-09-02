# ── text2sql-toolkit — Azure Container Apps Deployment ───────────────
#
# Deploys:
#   • Azure Container Registry (ACR)
#   • Container Apps Environment + Container App
#   • Azure Blob Storage for profile cache
#   • Key Vault for secrets
#   • Managed Identity with least-privilege
#   • Log Analytics workspace
#
# Usage:
#   terraform init
#   terraform plan -var="openai_api_key=sk-..."
#   terraform apply
# ─────────────────────────────────────────────────────────────────────

terraform {
  required_version = ">= 1.5"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.90"
    }
  }

  # Uncomment for remote state (recommended for teams)
  # backend "azurerm" {
  #   resource_group_name  = "terraform-state-rg"
  #   storage_account_name = "tfstatetext2sql"
  #   container_name       = "tfstate"
  #   key                  = "text2sql.terraform.tfstate"
  # }
}

provider "azurerm" {
  features {
    key_vault {
      purge_soft_delete_on_destroy = true
    }
  }
}

# ── Variables ────────────────────────────────────────────────────────

variable "azure_location" {
  description = "Azure region for deployment"
  type        = string
  default     = "East US"
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

variable "container_cpu" {
  description = "CPU cores for the container (e.g. 0.5, 1, 2)"
  type        = number
  default     = 1
}

variable "container_memory" {
  description = "Memory in Gi for the container (e.g. '1Gi', '2Gi')"
  type        = string
  default     = "2Gi"
}

variable "min_replicas" {
  description = "Minimum container replicas (0 = scale to zero)"
  type        = number
  default     = 0
}

variable "max_replicas" {
  description = "Maximum container replicas"
  type        = number
  default     = 10
}

variable "db_uri" {
  description = "Database URI for the target database"
  type        = string
  default     = "sqlite:///example.db"
  sensitive   = true
}

locals {
  prefix       = "${var.project_name}-${var.environment}"
  prefix_clean = replace(local.prefix, "-", "")
}

data "azurerm_client_config" "current" {}

# ── Resource Group ──────────────────────────────────────────────────

resource "azurerm_resource_group" "text2sql" {
  name     = "${local.prefix}-rg"
  location = var.azure_location

  tags = {
    Project     = "text2sql-toolkit"
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# ── Azure Container Registry ───────────────────────────────────────

resource "azurerm_container_registry" "text2sql" {
  name                = "${local.prefix_clean}acr"
  resource_group_name = azurerm_resource_group.text2sql.name
  location            = azurerm_resource_group.text2sql.location
  sku                 = "Basic"
  admin_enabled       = true
}

# ── Storage Account (Profile Cache) ────────────────────────────────

resource "azurerm_storage_account" "profile_cache" {
  name                     = "${local.prefix_clean}cache"
  resource_group_name      = azurerm_resource_group.text2sql.name
  location                 = azurerm_resource_group.text2sql.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  min_tls_version          = "TLS1_2"

  blob_properties {
    versioning_enabled = true
  }
}

resource "azurerm_storage_container" "profiles" {
  name                  = "profiles"
  storage_account_name  = azurerm_storage_account.profile_cache.name
  container_access_type = "private"
}

# ── Key Vault (Secrets) ────────────────────────────────────────────

resource "azurerm_key_vault" "text2sql" {
  name                = "${local.prefix_clean}kv"
  resource_group_name = azurerm_resource_group.text2sql.name
  location            = azurerm_resource_group.text2sql.location
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"

  purge_protection_enabled   = false
  soft_delete_retention_days = 7

  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = data.azurerm_client_config.current.object_id

    secret_permissions = [
      "Get", "List", "Set", "Delete", "Purge",
    ]
  }
}

resource "azurerm_key_vault_secret" "openai_key" {
  name         = "openai-api-key"
  value        = var.openai_api_key
  key_vault_id = azurerm_key_vault.text2sql.id
}

# ── Log Analytics ──────────────────────────────────────────────────

resource "azurerm_log_analytics_workspace" "text2sql" {
  name                = "${local.prefix}-logs"
  resource_group_name = azurerm_resource_group.text2sql.name
  location            = azurerm_resource_group.text2sql.location
  sku                 = "PerGB2018"
  retention_in_days   = var.environment == "prod" ? 90 : 30
}

# ── Container Apps Environment ─────────────────────────────────────

resource "azurerm_container_app_environment" "text2sql" {
  name                       = "${local.prefix}-env"
  resource_group_name        = azurerm_resource_group.text2sql.name
  location                   = azurerm_resource_group.text2sql.location
  log_analytics_workspace_id = azurerm_log_analytics_workspace.text2sql.id
}

# ── Managed Identity ───────────────────────────────────────────────

resource "azurerm_user_assigned_identity" "text2sql" {
  name                = "${local.prefix}-identity"
  resource_group_name = azurerm_resource_group.text2sql.name
  location            = azurerm_resource_group.text2sql.location
}

# Grant Key Vault access to managed identity
resource "azurerm_key_vault_access_policy" "container_app" {
  key_vault_id = azurerm_key_vault.text2sql.id
  tenant_id    = data.azurerm_client_config.current.tenant_id
  object_id    = azurerm_user_assigned_identity.text2sql.principal_id

  secret_permissions = ["Get", "List"]
}

# Grant Storage access to managed identity
resource "azurerm_role_assignment" "storage_blob" {
  scope                = azurerm_storage_account.profile_cache.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_user_assigned_identity.text2sql.principal_id
}

# Grant ACR pull to managed identity
resource "azurerm_role_assignment" "acr_pull" {
  scope                = azurerm_container_registry.text2sql.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_user_assigned_identity.text2sql.principal_id
}

# ── Container App ──────────────────────────────────────────────────

resource "azurerm_container_app" "text2sql" {
  name                         = local.prefix
  container_app_environment_id = azurerm_container_app_environment.text2sql.id
  resource_group_name          = azurerm_resource_group.text2sql.name
  revision_mode                = "Single"

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.text2sql.id]
  }

  registry {
    server   = azurerm_container_registry.text2sql.login_server
    identity = azurerm_user_assigned_identity.text2sql.id
  }

  ingress {
    external_enabled = true
    target_port      = 8000
    transport        = "auto"

    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  template {
    min_replicas = var.min_replicas
    max_replicas = var.max_replicas

    container {
      name   = "text2sql"
      image  = "${azurerm_container_registry.text2sql.login_server}/${var.project_name}:latest"
      cpu    = var.container_cpu
      memory = var.container_memory

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
        value = ".cache/profiles"
      }

      env {
        name  = "TEXT2SQL_LOG_LEVEL"
        value = var.environment == "prod" ? "WARNING" : "INFO"
      }

      env {
        name        = "OPENAI_API_KEY"
        secret_name = "openai-api-key"
      }

      liveness_probe {
        transport = "HTTP"
        path      = "/health"
        port      = 8000

        initial_delay    = 10
        interval_seconds = 30
      }

      readiness_probe {
        transport = "HTTP"
        path      = "/health"
        port      = 8000

        interval_seconds = 10
      }
    }
  }

  secret {
    name                = "openai-api-key"
    key_vault_secret_id = azurerm_key_vault_secret.openai_key.id
    identity            = azurerm_user_assigned_identity.text2sql.id
  }

  depends_on = [
    azurerm_key_vault_access_policy.container_app,
    azurerm_role_assignment.acr_pull,
  ]
}

# ── Outputs ─────────────────────────────────────────────────────────

output "acr_login_server" {
  description = "ACR login server for pushing Docker images"
  value       = azurerm_container_registry.text2sql.login_server
}

output "container_app_url" {
  description = "Container App FQDN"
  value       = "https://${azurerm_container_app.text2sql.ingress[0].fqdn}"
}

output "resource_group" {
  description = "Resource group name"
  value       = azurerm_resource_group.text2sql.name
}

output "storage_account" {
  description = "Storage account for profile cache"
  value       = azurerm_storage_account.profile_cache.name
}

output "key_vault_name" {
  description = "Key Vault name"
  value       = azurerm_key_vault.text2sql.name
}

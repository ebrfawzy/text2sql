# ── text2sql-toolkit — AWS Serverless Deployment ─────────────────────
#
# Deploys:
#   • ECR repository for container images
#   • Lambda function (container-based) with Function URL
#   • API Gateway v2 (HTTP API) with SSE streaming support
#   • S3 bucket for profile cache
#   • IAM roles and policies
#   • CloudWatch log groups
#
# Usage:
#   terraform init
#   terraform plan -var="openai_api_key=sk-..."
#   terraform apply
# ─────────────────────────────────────────────────────────────────────

terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Uncomment for remote state (recommended for teams)
  # backend "s3" {
  #   bucket = "your-terraform-state-bucket"
  #   key    = "text2sql/terraform.tfstate"
  #   region = "us-east-1"
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "text2sql-toolkit"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# ── Variables ────────────────────────────────────────────────────────

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
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

variable "lambda_memory_mb" {
  description = "Lambda memory allocation in MB"
  type        = number
  default     = 1024
}

variable "lambda_timeout_seconds" {
  description = "Lambda timeout in seconds (max 900)"
  type        = number
  default     = 900
}

variable "lambda_architecture" {
  description = "Lambda CPU architecture (x86_64 or arm64)"
  type        = string
  default     = "x86_64"
}

variable "enable_vpc" {
  description = "Deploy Lambda inside a VPC for private DB access"
  type        = bool
  default     = false
}

variable "vpc_id" {
  description = "VPC ID (required if enable_vpc = true)"
  type        = string
  default     = ""
}

variable "subnet_ids" {
  description = "Subnet IDs for Lambda (required if enable_vpc = true)"
  type        = list(string)
  default     = []
}

variable "db_uri" {
  description = "Database URI for the target database"
  type        = string
  default     = "sqlite:///example.db"
  sensitive   = true
}

locals {
  prefix = "${var.project_name}-${var.environment}"
}

# ── ECR Repository ──────────────────────────────────────────────────

resource "aws_ecr_repository" "text2sql" {
  name                 = "${local.prefix}-lambda"
  image_tag_mutability = "MUTABLE"
  force_delete         = var.environment != "prod"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

resource "aws_ecr_lifecycle_policy" "text2sql" {
  repository = aws_ecr_repository.text2sql.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 5 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 5
      }
      action = {
        type = "expire"
      }
    }]
  })
}

# ── S3 Profile Cache ────────────────────────────────────────────────

resource "aws_s3_bucket" "profile_cache" {
  bucket        = "${local.prefix}-profile-cache"
  force_destroy = var.environment != "prod"
}

resource "aws_s3_bucket_versioning" "profile_cache" {
  bucket = aws_s3_bucket.profile_cache.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "profile_cache" {
  bucket = aws_s3_bucket.profile_cache.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "profile_cache" {
  bucket                  = aws_s3_bucket.profile_cache.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ── IAM Role for Lambda ─────────────────────────────────────────────

resource "aws_iam_role" "lambda" {
  name = "${local.prefix}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "lambda_s3" {
  name = "${local.prefix}-lambda-s3"
  role = aws_iam_role.lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:GetObject",
        "s3:PutObject",
        "s3:HeadObject",
        "s3:ListBucket",
      ]
      Resource = [
        aws_s3_bucket.profile_cache.arn,
        "${aws_s3_bucket.profile_cache.arn}/*",
      ]
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy_attachment" "lambda_vpc" {
  count      = var.enable_vpc ? 1 : 0
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole"
}

# ── Security Group (VPC mode) ───────────────────────────────────────

resource "aws_security_group" "lambda" {
  count       = var.enable_vpc ? 1 : 0
  name        = "${local.prefix}-lambda-sg"
  description = "Security group for text2sql Lambda"
  vpc_id      = var.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound (LLM APIs + DB)"
  }
}

# ── CloudWatch Log Group ────────────────────────────────────────────

resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/${local.prefix}"
  retention_in_days = var.environment == "prod" ? 90 : 14
}

# ── Lambda Function ─────────────────────────────────────────────────

resource "aws_lambda_function" "text2sql" {
  function_name = local.prefix
  role          = aws_iam_role.lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.text2sql.repository_url}:latest"
  architectures = [var.lambda_architecture]
  memory_size   = var.lambda_memory_mb
  timeout       = var.lambda_timeout_seconds

  environment {
    variables = {
      TEXT2SQL_MODEL             = var.llm_model
      TEXT2SQL_DB_URI            = var.db_uri
      TEXT2SQL_PROFILE_CACHE_DIR = "s3://${aws_s3_bucket.profile_cache.bucket}/profiles"
      TEXT2SQL_LOG_LEVEL         = var.environment == "prod" ? "WARNING" : "INFO"
      OPENAI_API_KEY            = var.openai_api_key
    }
  }

  dynamic "vpc_config" {
    for_each = var.enable_vpc ? [1] : []
    content {
      subnet_ids         = var.subnet_ids
      security_group_ids = [aws_security_group.lambda[0].id]
    }
  }

  depends_on = [
    aws_cloudwatch_log_group.lambda,
    aws_iam_role_policy_attachment.lambda_basic,
  ]
}

# ── Lambda Function URL (simple direct access) ─────────────────────

resource "aws_lambda_function_url" "text2sql" {
  function_name      = aws_lambda_function.text2sql.function_name
  authorization_type = "NONE"

  invoke_mode = "RESPONSE_STREAM"

  cors {
    allow_origins = ["*"]
    allow_methods = ["POST"]
    allow_headers = ["Content-Type"]
    max_age       = 3600
  }
}

# ── API Gateway v2 (HTTP API) ───────────────────────────────────────

resource "aws_apigatewayv2_api" "text2sql" {
  name          = "${local.prefix}-api"
  protocol_type = "HTTP"

  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["GET", "POST", "OPTIONS"]
    allow_headers = ["Content-Type", "Authorization"]
    max_age       = 3600
  }
}

resource "aws_apigatewayv2_integration" "lambda" {
  api_id                 = aws_apigatewayv2_api.text2sql.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.text2sql.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "default" {
  api_id    = aws_apigatewayv2_api.text2sql.id
  route_key = "$default"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.text2sql.id
  name        = "$default"
  auto_deploy = true

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_gateway.arn
    format = jsonencode({
      requestId      = "$context.requestId"
      ip             = "$context.identity.sourceIp"
      requestTime    = "$context.requestTime"
      httpMethod     = "$context.httpMethod"
      routeKey       = "$context.routeKey"
      status         = "$context.status"
      responseLength = "$context.responseLength"
      integrationError = "$context.integrationErrorMessage"
    })
  }
}

resource "aws_cloudwatch_log_group" "api_gateway" {
  name              = "/aws/apigateway/${local.prefix}"
  retention_in_days = 14
}

resource "aws_lambda_permission" "api_gateway" {
  statement_id  = "AllowAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.text2sql.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.text2sql.execution_arn}/*/*"
}

# ── Outputs ─────────────────────────────────────────────────────────

output "ecr_repository_url" {
  description = "ECR repository URL for pushing Docker images"
  value       = aws_ecr_repository.text2sql.repository_url
}

output "lambda_function_name" {
  description = "Lambda function name"
  value       = aws_lambda_function.text2sql.function_name
}

output "lambda_function_url" {
  description = "Direct Lambda Function URL (streaming support)"
  value       = aws_lambda_function_url.text2sql.function_url
}

output "api_gateway_url" {
  description = "API Gateway endpoint URL"
  value       = aws_apigatewayv2_stage.default.invoke_url
}

output "s3_profile_cache" {
  description = "S3 bucket for profile cache"
  value       = "s3://${aws_s3_bucket.profile_cache.bucket}/profiles"
}

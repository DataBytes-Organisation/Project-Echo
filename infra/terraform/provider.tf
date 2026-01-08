terraform {
  required_version = ">= 1.6.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.30"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.30"
    }
  }
}

provider "google" {
  project               = var.project_id
  region                = var.region
  user_project_override = true
  # Provide credentials via GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_CLOUD_KEYFILE_JSON env vars.
}

provider "google-beta" {
  project               = var.project_id
  region                = var.region
  user_project_override = true
  # Provide credentials via GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_CLOUD_KEYFILE_JSON env vars.
}

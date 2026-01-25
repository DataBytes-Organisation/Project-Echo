locals {
  pod_range_name     = "${var.subnet_name}-pods"
  service_range_name = "${var.subnet_name}-services"
  router_name        = "${var.network_name}-router"
  nat_name           = "${var.network_name}-nat"
}

resource "google_compute_network" "this" {
  name                    = var.network_name
  project                 = var.project_id
  auto_create_subnetworks = false
  routing_mode            = "REGIONAL"
}

resource "google_compute_subnetwork" "primary" {
  name          = var.subnet_name
  project       = var.project_id
  region        = var.region
  network       = google_compute_network.this.id
  ip_cidr_range = var.primary_cidr
  stack_type    = "IPV4_ONLY"

  secondary_ip_range {
    range_name    = local.pod_range_name
    ip_cidr_range = var.pod_cidr
  }

  secondary_ip_range {
    range_name    = local.service_range_name
    ip_cidr_range = var.service_cidr
  }
}

resource "google_compute_router" "this" {
  name    = local.router_name
  project = var.project_id
  region  = var.region
  network = google_compute_network.this.id
}

resource "google_compute_router_nat" "this" {
  name                               = local.nat_name
  project                            = var.project_id
  region                             = var.region
  router                             = google_compute_router.this.name
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    filter = "ERRORS_ONLY"
    enable = true
  }
}

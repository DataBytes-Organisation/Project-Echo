#!/usr/bin/env bash
set -euo pipefail

# gcp-echo-discover.sh — discover Echo/EchoNet endpoints and status on GCP
# Usage: bash scripts/gcp-echo-discover.sh [namespace-regex] [name-regex]
# Defaults search for "echo|echonet" across GKE and Cloud Run.

NS_RE="${1:-.*}"
NAME_RE="${2:-echo|echonet}"
PROJECT="$(gcloud config get-value project 2>/dev/null)"
TS() { date +"%Y-%m-%dT%H:%M:%S%z"; }

printf "[%s] Project: %s\n" "$(TS)" "${PROJECT}"

found_any=false

# ---- Cloud Run ----
if gcloud services list --enabled --format=value(config.name) | grep -q '^run.googleapis.com$'; then
  printf "\n[%s] Checking Cloud Run services...\n" "$(TS)"
  while read -r REGION; do
    [ -z "$REGION" ] && continue
    SRV=$(gcloud run services list --region "$REGION" --format='value(name, status.url)' || true)
    if [[ -n "$SRV" ]]; then
      echo "$SRV" | awk -v re="$NAME_RE" '$1 ~ re {print $1 "\t" $2}' | while IFS=$'\t' read -r NAME URL; do
        found_any=true
        printf "Cloud Run | region=%s | service=%s | url=%s\n" "$REGION" "$NAME" "$URL"
        # Health guesses
        for path in /healthz /health /readyz /; do
          code=$(curl -sk -o /dev/null -w '%{http_code}' "${URL%/}${path}") || code=000
          printf "  -> %s : HTTP %s\n" "$path" "$code"
          [ "$code" = "200" ] && break
        done
      done
    fi
  done < <(gcloud run regions list --format='value(name)')
else
  printf "\n[%s] Cloud Run API not enabled or no access.\n" "$(TS)"
fi

# ---- GKE ----
if gcloud services list --enabled --format=value(config.name) | grep -q '^container.googleapis.com$'; then
  printf "\n[%s] Checking GKE clusters...\n" "$(TS)"
  while read -r CLUSTER LOCATION; do
    [ -z "$CLUSTER" ] && continue
    printf "Cluster: %s (%s)\n" "$CLUSTER" "$LOCATION"
    gcloud container clusters get-credentials "$CLUSTER" --region "$LOCATION" >/dev/null 2>&1 || \
    gcloud container clusters get-credentials "$CLUSTER" --zone "$LOCATION" >/dev/null 2>&1 || true

    # Namespaces matching pattern
    kubectl get ns --no-headers 2>/dev/null | awk -v re="$NS_RE" '$1 ~ re {print $1}' | while read -r ns; do
      # Deployments with name filter
      kubectl -n "$ns" get deploy --no-headers 2>/dev/null | awk -v re="$NAME_RE" '$1 ~ re {print $1}' | while read -r app; do
        found_any=true
        img=$(kubectl -n "$ns" get deploy "$app" -o jsonpath='{.spec.template.spec.containers[0].image}' 2>/dev/null || echo -n)
        avail=$(kubectl -n "$ns" get deploy "$app" -o jsonpath='{.status.conditions[?(@.type=="Available")].lastUpdateTime}' 2>/dev/null || echo -n)
        printf "GKE | ns=%s | app=%s | image=%s | availableAt=%s\n" "$ns" "$app" "$img" "$avail"
      done

      # Ingress hosts
      kubectl -n "$ns" get ingress --no-headers 2>/dev/null | awk '{print $1}' | while read -r ing; do
        host=$(kubectl -n "$ns" get ingress "$ing" -o jsonpath='{.spec.rules[*].host}' 2>/dev/null || echo -n)
        [ -z "$host" ] && continue
        printf "  Ingress: %s | host=%s\n" "$ing" "$host"
        for path in /healthz /health /readyz /; do
          code=$(curl -sk -o /dev/null -w '%{http_code}' "https://${host%/}${path}") || code=000
          printf "    -> %s : HTTP %s\n" "$path" "$code"
          [ "$code" = "200" ] && break
        done
      done

      # LoadBalancer services
      kubectl -n "$ns" get svc --no-headers 2>/dev/null | awk '$3=="LoadBalancer" {print $1}' | while read -r svc; do
        ip=$(kubectl -n "$ns" get svc "$svc" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo -n)
        host=$(kubectl -n "$ns" get svc "$svc" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo -n)
        target="${host:-$ip}"
        [ -z "$target" ] && continue
        printf "  SVC: %s | external=%s\n" "$svc" "$target"
        for path in /healthz /health /readyz /; do
          code=$(curl -sk -o /dev/null -w '%{http_code}' "http://${target%/}${path}") || code=000
          printf "    -> %s : HTTP %s\n" "$path" "$code"
          [ "$code" = "200" ] && break
        done
      done
    done
  done < <(gcloud container clusters list --format='value(name,location)')
else
  printf "\n[%s] GKE API not enabled or no access.\n" "$(TS)"
fi

# ---- Fallback: load balancers / forwarding rules ----
printf "\n[%s] Checking external forwarding rules (HTTP(S))...\n" "$(TS)"
gcloud compute forwarding-rules list --filter='target~https|http' --format='table(name, IPAddress, target)' || true

if ! $found_any; then
  printf "\nNo Echo/EchoNet endpoints found yet. Try adjusting search: bash scripts/gcp-echo-discover.sh '.*' 'your-app-name'\n"
fi

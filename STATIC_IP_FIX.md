# Fix Dynamic IP Issue - Reserve Static IP for GKE

## Problem
Your external IP keeps changing because you're using an ephemeral (dynamic) IP address.
- Old IP: 34.151.181.137 (no longer works)
- Current IP: 34.151.179.104 (will change if service restarts)

## Solution: Reserve a Static IP

### Step 1: Reserve a Static IP Address
```powershell
# Reserve a static external IP in your region
gcloud compute addresses create project-echo-static-ip `
  --region=australia-southeast1 `
  --network-tier=PREMIUM

# Get the reserved IP address
gcloud compute addresses describe project-echo-static-ip `
  --region=australia-southeast1 `
  --format="get(address)"
```

### Step 2: Update Your Kubernetes Service

Edit your HMI service to use the static IP:

**Option A: Using kubectl (if plugin is installed)**
```powershell
kubectl annotate service hmi-service `
  "cloud.google.com/load-balancer-type=external" `
  --overwrite `
  -n default

kubectl patch service hmi-service `
  -p '{"spec":{"loadBalancerIP":"YOUR_STATIC_IP_HERE"}}' `
  -n default
```

**Option B: Update the YAML file directly**

Edit [k8s/base/hmi-service.yaml](k8s/base/hmi-service.yaml):

```yaml
apiVersion: v1
kind: Service
metadata:
  name: hmi-service
  namespace: default
spec:
  type: LoadBalancer
  loadBalancerIP: "34.151.XXX.XXX"  # Add your static IP here
  ports:
    - name: http
      port: 8080
      targetPort: 3000
      protocol: TCP
  selector:
    app: hmi
```

Then reapply:
```powershell
kubectl apply -f k8s/base/hmi-service.yaml
```

### Step 3: Verify Static IP is Assigned
```powershell
# Check service external IP
gcloud compute forwarding-rules list

# Or using kubectl
kubectl get service hmi-service -n default
```

## Alternative: Use a Domain Name

Instead of relying on IP addresses, set up a domain name:

### Option 1: Use Google Cloud DNS
```powershell
# Create a DNS zone
gcloud dns managed-zones create project-echo `
  --dns-name="yourdomain.com." `
  --description="Project Echo DNS"

# Add A record pointing to your static IP
gcloud dns record-sets create login.yourdomain.com. `
  --zone="project-echo" `
  --type="A" `
  --ttl="300" `
  --rrdatas="34.151.179.104"
```

### Option 2: Update Ingress (Recommended for Production)

If using Ingress (which you have configured), you can:

1. Reserve a static IP for Ingress:
```powershell
gcloud compute addresses create project-echo-ingress-ip `
  --global `
  --ip-version=IPV4
```

2. Update [k8s/base/ingress.yaml](k8s/base/ingress.yaml):
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: project-echo
  namespace: project-echo
  annotations:
    kubernetes.io/ingress.class: "gce"
    kubernetes.io/ingress.global-static-ip-name: "project-echo-ingress-ip"
    networking.gke.io/managed-certificates: project-echo-cert
spec:
  rules:
    - host: echo.yourdomain.com  # Use your domain
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: hmi
                port:
                  name: http
```

## Quick Command Reference

```powershell
# List all static IPs
gcloud compute addresses list

# Delete old ephemeral forwarding rule (if needed)
gcloud compute forwarding-rules delete a36876cb6efa64644875403f385a02b4 --region=australia-southeast1

# Check current service details
kubectl get service hmi-service -n default -o yaml
```

## Cost Note
Static IP addresses in GCP are:
- **FREE** when attached to a running resource (like your load balancer)
- **~$3/month** when reserved but not in use
- Much cheaper than dealing with IP changes!

## Recommended Next Steps
1. Reserve a static IP immediately (5 minutes)
2. Update your service to use it (10 minutes)
3. Consider setting up a proper domain name for production use
4. Update your documentation with the static IP address

# Deploy Login Fix to GKE (34.151.181.137)

## Quick Summary
Your remote server `34.151.181.137` is running on **Google Kubernetes Engine (GKE)**. The login works on localhost but not on GKE because:

1. **Code is not deployed** - The fixes are only on your local machine
2. **reCAPTCHA domain** - IP address not in allowed domains

## Fastest Fix (5 minutes) - Add reCAPTCHA Domain

This will make login work **immediately** without code deployment:

### Step 1: Add IP to reCAPTCHA Console
1. Go to https://console.cloud.google.com/security/recaptcha
2. Select the site key: `6Lee1k0sAAAAAGYhpdP_0jcaghmD5Ta6K8WPUsyA`
3. Click "Settings" or edit icon
4. Under "Domains", add:
   ```
   34.151.181.137
   ```
5. Click "Save"
6. Wait 1-2 minutes for propagation

### Step 2: Test Login
1. Open http://34.151.181.137:8080/login (use incognito mode)
2. Enter credentials
3. Login should work now!

---

## Proper Fix - Deploy Updated Code to GKE

### Prerequisites
```powershell
# Install Google Cloud SDK if not already installed
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Get cluster credentials
gcloud container clusters get-credentials echonet-gke --region australia-southeast2
```

### Option 1: Deploy via Cloud Build (Automated)

```powershell
# From your local project directory
cd C:\Users\syyen_ybpva\Project-Echo

# Commit your changes
git add src/Echo_Components_on_K8s/frontend/routes/auth.routes.js
git add src/Echo_Components_on_K8s/frontend/public/login.html
git commit -m "Fix login authentication and reCAPTCHA"
git push origin main

# Trigger Cloud Build
gcloud builds submit --config=cloudbuild.yaml .

# This will:
# 1. Build new Docker images with your fixes
# 2. Push to Google Container Registry
# 3. Deploy to GKE
```

### Option 2: Manual Docker Build & Push

```powershell
cd C:\Users\syyen_ybpva\Project-Echo

# Set variables
$PROJECT_ID = "YOUR_PROJECT_ID"
$REGION = "australia-southeast2"
$REPOSITORY = "echonet"

# Build the frontend image with fixes
cd src\Echo_Components_on_K8s\frontend
docker build -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/hmi:fixed .

# Push to registry
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/hmi:fixed

# Update Kubernetes deployment
kubectl set image deployment/hmi-deployment hmi=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/hmi:fixed -n echo

# Check rollout status
kubectl rollout status deployment/hmi-deployment -n echo
```

### Option 3: Quick Patch (Update ConfigMap)

If the login HTML is in a ConfigMap:

```powershell
# Edit the ConfigMap
kubectl edit configmap hmi-config -n echo

# Update the login.html content with fixes
# Save and exit

# Restart pods to pick up changes
kubectl rollout restart deployment/hmi-deployment -n echo
```

---

## Verify Deployment

### Check Pod Status
```powershell
kubectl get pods -n echo
kubectl logs -f deployment/hmi-deployment -n echo
```

### Check Service
```powershell
kubectl get svc -n echo
kubectl describe svc hmi-service -n echo
```

### Check Ingress
```powershell
kubectl get ingress -n echo
kubectl describe ingress echo-ingress -n echo
```

### Test Login
```powershell
# Check if service is responding
Invoke-WebRequest -Uri "http://34.151.181.137:8080/login" -UseBasicParsing

# Test authentication endpoint
Invoke-WebRequest -Uri "http://34.151.181.137:8080/api/auth/signin" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"username":"TN-HMI","password":"your_password"}' `
  -UseBasicParsing
```

---

## Architecture Overview

```
Local Machine (localhost:8080) ✅ Working
    ↓ Fixed code exists here
    
GKE Cluster (34.151.181.137) ❌ Old code
    ├── HMI Frontend (Port 8080)
    ├── API Backend (ts-api-cont, Port 9000)
    ├── MongoDB (Port 27017)
    └── Redis (Port 6379)
```

---

## Files to Deploy

These files have the fixes and need to be in the Docker image:

1. `src/Echo_Components_on_K8s/frontend/routes/auth.routes.js`
   - Local MongoDB authentication fallback
   - Fixed database name

2. `src/Echo_Components_on_K8s/frontend/public/login.html`
   - Optional reCAPTCHA
   - Better error handling

---

## Dockerfile Check

Make sure the Dockerfile includes these files:

```dockerfile
# src/Echo_Components_on_K8s/frontend/Dockerfile
FROM node:18

WORKDIR /app

COPY package*.json ./
RUN npm install

# Make sure these are copied
COPY routes/ ./routes/
COPY public/ ./public/
COPY . .

EXPOSE 8080
CMD ["node", "server.js"]
```

---

## Rollback if Needed

If something goes wrong:

```powershell
# Rollback to previous version
kubectl rollout undo deployment/hmi-deployment -n echo

# Check rollout history
kubectl rollout history deployment/hmi-deployment -n echo

# Rollback to specific revision
kubectl rollout undo deployment/hmi-deployment -n echo --to-revision=2
```

---

## Environment Variables

Make sure these are set in your Kubernetes deployment:

```yaml
env:
  - name: MONGODB_URI
    value: "mongodb://root:root_password@mongodb-service:27017"
  - name: API_URL
    value: "http://ts-api-cont:9000"
  - name: NODE_ENV
    value: "production"
```

---

## Complete Deployment Script

Create this file: `deploy_fix.ps1`

```powershell
# Deploy Login Fix to GKE
$ErrorActionPreference = "Stop"

Write-Host "=== Deploying Login Fix to GKE ===" -ForegroundColor Green

# Set variables - UPDATE THESE
$PROJECT_ID = "your-project-id"
$REGION = "australia-southeast2"
$REPOSITORY = "echonet"
$CLUSTER = "echonet-gke"
$NAMESPACE = "echo"

# Authenticate
Write-Host "Authenticating..." -ForegroundColor Yellow
gcloud config set project $PROJECT_ID
gcloud container clusters get-credentials $CLUSTER --region $REGION

# Build and push
Write-Host "Building Docker image..." -ForegroundColor Yellow
cd src\Echo_Components_on_K8s\frontend
$IMAGE_TAG = Get-Date -Format "yyyyMMdd-HHmmss"
$IMAGE_NAME = "$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/hmi:$IMAGE_TAG"

docker build -t $IMAGE_NAME .
docker push $IMAGE_NAME

# Update deployment
Write-Host "Updating Kubernetes deployment..." -ForegroundColor Yellow
kubectl set image deployment/hmi-deployment hmi=$IMAGE_NAME -n $NAMESPACE

# Wait for rollout
Write-Host "Waiting for rollout..." -ForegroundColor Yellow
kubectl rollout status deployment/hmi-deployment -n $NAMESPACE

# Verify
Write-Host "Verifying pods..." -ForegroundColor Yellow
kubectl get pods -n $NAMESPACE | Select-String "hmi"

Write-Host "`n=== Deployment Complete! ===" -ForegroundColor Green
Write-Host "Test login at: http://34.151.181.137:8080/login" -ForegroundColor Cyan
```

Run it:
```powershell
cd C:\Users\syyen_ybpva\Project-Echo
.\deploy_fix.ps1
```

---

## Quick Test After Deployment

```powershell
# Test 1: Check if login page loads
Invoke-WebRequest -Uri "http://34.151.181.137:8080/login" -UseBasicParsing

# Test 2: Check pods are running
kubectl get pods -n echo | Select-String "hmi"

# Test 3: Check logs
kubectl logs -f deployment/hmi-deployment -n echo --tail=50

# Test 4: Try to login via browser
Start-Process "http://34.151.181.137:8080/login"
```

---

## Troubleshooting

### If deployment fails:

**Check build logs:**
```powershell
gcloud builds list --limit 5
gcloud builds log [BUILD_ID]
```

**Check pod logs:**
```powershell
kubectl get pods -n echo
kubectl logs [POD_NAME] -n echo
kubectl describe pod [POD_NAME] -n echo
```

**Check service:**
```powershell
kubectl get svc -n echo
kubectl port-forward svc/hmi-service 8080:8080 -n echo
# Then test: http://localhost:8080/login
```

---

## Summary

**Recommended approach:**

1. **Immediate fix (5 min):** Add `34.151.181.137` to reCAPTCHA domains
2. **Proper fix (30 min):** Deploy updated code to GKE

**After deployment:**
- ✅ Login works on both localhost AND remote server
- ✅ reCAPTCHA doesn't block users
- ✅ Authentication works with or without external API

Need help with deployment? Let me know:
- Your GCP project ID
- Whether you have gcloud CLI installed
- If you can access the GKE cluster

# Deploy Login Fixes to Remote Server (34.151.181.137)

## Problem
You can login on localhost but NOT on the remote server `http://34.151.181.137:8080/login`

## Root Cause
The code fixes I made are only on your **local machine**. The remote server at `34.151.181.137` still has the old code with:
1. reCAPTCHA blocking the form submission
2. Possible authentication issues

## Solution: Deploy Updated Files to Remote Server

### Option 1: SSH/SCP Deployment (Recommended)

#### Step 1: Connect to Remote Server
```powershell
# SSH into the server (replace with your actual credentials)
ssh your_username@34.151.181.137

# OR if using a key file
ssh -i path/to/key.pem your_username@34.151.181.137
```

#### Step 2: Copy Updated Files
From your **local machine**, copy the updated files:

```powershell
# Navigate to your project directory
cd C:\Users\syyen_ybpva\Project-Echo

# Copy the updated auth routes file
scp src\Components\HMI\ui\routes\auth.routes.js your_username@34.151.181.137:/path/to/remote/project/src/Components/HMI/ui/routes/

# Copy the updated login HTML file
scp src\Components\HMI\ui\public\login.html your_username@34.151.181.137:/path/to/remote/project/src/Components/HMI/ui/public/

# OR copy for Echo_Components_on_K8s if that's what's running on remote
scp src\Echo_Components_on_K8s\frontend\routes\auth.routes.js your_username@34.151.181.137:/path/to/remote/project/src/Echo_Components_on_K8s/frontend/routes/

scp src\Echo_Components_on_K8s\frontend\public\login.html your_username@34.151.181.137:/path/to/remote/project/src/Echo_Components_on_K8s/frontend/public/
```

#### Step 3: Restart Services on Remote Server
```bash
# SSH into the remote server
ssh your_username@34.151.181.137

# Navigate to the project directory
cd /path/to/remote/project

# Restart Docker containers
docker restart ts-api-cont
docker ps | grep ts-api

# If using docker-compose
docker-compose down
docker-compose up -d

# OR restart Node.js server
pm2 restart all
# OR
pkill -f "node server.js"
cd src/Components/HMI/ui
node server.js &
```

---

### Option 2: Git Deployment (If Using Version Control)

```powershell
# On your local machine - commit and push changes
cd C:\Users\syyen_ybpva\Project-Echo
git add src/Components/HMI/ui/routes/auth.routes.js
git add src/Components/HMI/ui/public/login.html
git add src/Echo_Components_on_K8s/frontend/routes/auth.routes.js
git add src/Echo_Components_on_K8s/frontend/public/login.html
git commit -m "Fix login authentication and reCAPTCHA issues"
git push origin main

# On remote server - pull changes
ssh your_username@34.151.181.137
cd /path/to/remote/project
git pull origin main

# Restart services
docker restart ts-api-cont
pm2 restart all
```

---

### Option 3: Manual Copy via RDP/FTP

If you have RDP access or FTP:

1. **Connect to the remote server** via Remote Desktop or FTP client
2. **Navigate to the project directory** on the remote server
3. **Replace these files** with the updated versions from your local machine:
   - `src/Components/HMI/ui/routes/auth.routes.js`
   - `src/Components/HMI/ui/public/login.html`
   - `src/Echo_Components_on_K8s/frontend/routes/auth.routes.js`
   - `src/Echo_Components_on_K8s/frontend/public/login.html`
4. **Restart the services**

---

### Option 4: Quick Fix - reCAPTCHA Domain Configuration

If you can't deploy code immediately, fix reCAPTCHA configuration:

#### Add IP to reCAPTCHA Allowed Domains:

1. Go to [Google Cloud Console - reCAPTCHA](https://console.cloud.google.com/security/recaptcha)
2. Sign in with the Google account that owns the reCAPTCHA key
3. Find the key: `6Lee1k0sAAAAAGYhpdP_0jcaghmD5Ta6K8WPUsyA`
4. Click "Settings" or "Edit"
5. Under "Domains", add:
   ```
   34.151.181.137
   ```
6. Save changes
7. Wait 1-2 minutes for changes to propagate
8. Try logging in again

#### OR Disable reCAPTCHA Temporarily:

On the remote server, edit the login.html file to completely remove reCAPTCHA:

```bash
# SSH into server
ssh your_username@34.151.181.137

# Edit the login file
nano /path/to/project/src/Components/HMI/ui/public/login.html

# Comment out or remove the reCAPTCHA script tags:
# <script src="https://www.google.com/recaptcha/enterprise.js?..."></script>

# Comment out reCAPTCHA execution code in the login form submit handler
```

---

## Verification Steps

After deploying, verify the fix:

### 1. Check Server is Running
```powershell
Invoke-WebRequest -Uri "http://34.151.181.137:8080/login" -UseBasicParsing
```

### 2. Test Login
1. Open browser (incognito/private mode recommended)
2. Go to `http://34.151.181.137:8080/login`
3. Open Developer Tools (F12)
4. Go to Console tab
5. Try to login with credentials
6. Check for errors

### 3. Check Docker Container
```bash
ssh your_username@34.151.181.137
docker ps | grep ts-api
docker logs ts-api-cont
```

---

## Files That Were Modified (Need to Deploy)

### Core Authentication Files:
1. ✅ `src/Components/HMI/ui/routes/auth.routes.js`
   - Added local MongoDB fallback authentication
   - Fixed database name to UserSample
   - Added API availability check

2. ✅ `src/Components/HMI/ui/public/login.html`
   - Made reCAPTCHA optional
   - Improved error handling
   - Better error messages

3. ✅ `src/Echo_Components_on_K8s/frontend/routes/auth.routes.js`
   - Same authentication fixes

4. ✅ `src/Echo_Components_on_K8s/frontend/public/login.html`
   - Same reCAPTCHA fixes

---

## Troubleshooting Remote Server

### If login still fails after deployment:

**Check Server Logs:**
```bash
ssh your_username@34.151.181.137

# Docker logs
docker logs ts-api-cont --tail 50

# Node.js logs (if using PM2)
pm2 logs

# System logs
journalctl -u your-service-name -n 50
```

**Check Backend API:**
```bash
curl http://localhost:9000/
docker ps | grep ts-api
netstat -tulpn | grep 9000
```

**Check MongoDB:**
```bash
docker ps | grep mongo
docker exec -it mongo-container mongo -u root -p root_password
> use UserSample
> db.users.find({username: "TN-HMI"})
```

**Check Environment Variables:**
```bash
# On remote server
cd /path/to/project
cat .env
docker exec ts-api-cont env | grep MONGODB
```

---

## Alternative: Use Docker Image with Fixes

If you want a clean deployment:

```bash
# On your local machine - build Docker image
cd C:\Users\syyen_ybpva\Project-Echo
docker build -t project-echo-frontend:fixed .
docker tag project-echo-frontend:fixed your-registry/project-echo-frontend:fixed
docker push your-registry/project-echo-frontend:fixed

# On remote server - pull and run
ssh your_username@34.151.181.137
docker pull your-registry/project-echo-frontend:fixed
docker stop frontend-container
docker run -d --name frontend-container -p 8080:8080 your-registry/project-echo-frontend:fixed
```

---

## Summary

**The key issue:** Your local code has the fixes, but the remote server doesn't.

**Quick Solution:**
1. Add `34.151.181.137` to reCAPTCHA allowed domains in Google Cloud Console
2. Wait 1-2 minutes
3. Try login again

**Proper Solution:**
1. Deploy updated `auth.routes.js` and `login.html` files to remote server
2. Restart services
3. Test login

**Need Help?**
Provide me with:
- How the remote server is hosted (Docker, K8s, VM?)
- How you typically deploy (Git, FTP, SCP?)
- Access method (SSH, RDP, cloud console?)

I can then provide specific deployment commands for your setup.

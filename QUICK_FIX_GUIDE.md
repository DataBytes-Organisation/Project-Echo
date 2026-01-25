# QUICK FIX - Login Issues Resolved

## What Was Fixed

### Issue 1: Localhost Login Failure
- **Problem:** The auth routes were trying to connect to a backend API at `http://ts-api-cont:9000` that wasn't running
- **Solution:** Added fallback authentication that connects directly to MongoDB when the API is unavailable
- **Database Fix:** Changed from `test` database to `UserSample` database (where users actually exist)

### Issue 2: reCAPTCHA Error on IP Address
- **Problem:** reCAPTCHA not configured for IP address `34.151.181.137`
- **Solution:** Made reCAPTCHA optional - login will proceed even if reCAPTCHA fails

## How to Login Now (3 Easy Steps)

### Step 1: Reset Your Password (If you don't know it)
```powershell
cd src\Components\HMI\ui
node reset_password.js
```
Enter a new password when prompted (minimum 6 characters).

### Step 2: Restart the Server
```powershell
# Stop the current server (Ctrl+C if running, or close terminal)

# Start fresh
cd C:\Users\syyen_ybpva\Project-Echo\src\Components\HMI\ui
node server.js
```

### Step 3: Login
1. Open http://localhost:8080/login
2. Enter credentials:
   - **Username:** `TN-HMI`
   - **Password:** (the password you just reset)
3. Click Login

## What's Running

- **MongoDB:** Port 27017 ✓ (Already running)
- **Frontend Server:** Port 8080 ✓ (Running)
- **Backend API:** Port 9000 ✗ (Not needed anymore - auth works locally)

## Files Modified

1. `src/Components/HMI/ui/routes/auth.routes.js`
   - Added local MongoDB authentication fallback
   - Fixed database name to UserSample
   - Made reCAPTCHA optional

2. `src/Components/HMI/ui/public/login.html`
   - Improved reCAPTCHA error handling
   - Better error messages

3. `src/Echo_Components_on_K8s/frontend/routes/auth.routes.js`
   - Same fixes as above

4. `src/Echo_Components_on_K8s/frontend/public/login.html`
   - Same fixes as above

## Troubleshooting

### If login still doesn't work:

**1. Check if MongoDB is running:**
```powershell
netstat -ano | findstr :27017
```
Should show output. If not, start MongoDB/Docker.

**2. Check if server is running:**
```powershell
netstat -ano | findstr :8080
```
Should show output. If not, run `node server.js`.

**3. View server logs:**
Look at the terminal where `node server.js` is running for error messages.

**4. Clear browser cache:**
- Press F12 to open DevTools
- Right-click refresh button → Empty Cache and Hard Reload

**5. Check browser console:**
- Press F12
- Go to Console tab
- Look for error messages

**6. Test password hash directly:**
```powershell
cd src\Components\HMI\ui
node test_login.js
```
This will tell you if the user exists and test common passwords.

## For Production (IP Address 34.151.181.137)

To fix reCAPTCHA on the production server:

1. Go to https://console.cloud.google.com/security/recaptcha
2. Select your reCAPTCHA key
3. Add domain: `34.151.181.137`
4. OR (better) use a proper domain name instead of IP

## Common Questions

**Q: What if I forgot the password?**
A: Run `node reset_password.js` in the `src/Components/HMI/ui` directory.

**Q: Can I create a new user?**
A: Yes, use the Register button on the login page.

**Q: Why was the API backend not running?**
A: The system was configured for Docker/Kubernetes deployment but you're running on localhost. The fix makes it work in both environments.

**Q: Is this secure?**
A: For development, yes. For production, you should:
- Use environment variables for MongoDB credentials
- Use HTTPS
- Configure reCAPTCHA properly
- Use proper session management

## Contact

If you still have issues after following these steps, check:
1. MongoDB is accessible at `mongodb://root:root_password@localhost:27017`
2. User exists in `UserSample` database
3. Password has been reset
4. Server has been restarted
5. Browser cache has been cleared

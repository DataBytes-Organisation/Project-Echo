## Secret Management (EchoNet)

Do not commit real secrets. Use this guide for local dev and deployment.

### Local Development
1. Create `secrets/mongo.env` (git-ignored):
   ```bash
   MONGODB_USER=your_user
   MONGODB_PASS=your_pass
   MONGODB_HOST=localhost
   MONGODB_DB=EchoNet
   ```
2. Load it: `export $(grep -v '^#' secrets/mongo.env | xargs)`
3. Run the API normally.

### Kubernetes
1. Create secret from sanitized manifest:
   ```bash
   kubectl apply -f deploy/helm/echonet/templates/mongo-credentials-secret.yaml -n <ns>
   ```
2. Or create directly:
   ```bash
   kubectl create secret generic mongo-credentials \
     --from-literal=MONGODB_USER=xxx \
     --from-literal=MONGODB_PASS=yyy \
     --from-literal=MONGODB_HOST=mongodb \
     --from-literal=MONGODB_DB=EchoNet -n <ns>
   ```
3. Reference in Deployment env:
   ```yaml
   envFrom:
     - secretRef:
         name: mongo-credentials
   ```

### Rotation
1. Add new Mongo user & password.
2. Update K8s Secret (kubectl apply -f or kubectl create secret ... --dry-run=client -o yaml | kubectl apply -f -).
3. Restart deployments (or rely on rolling update).
4. Remove old user after validation.

### Auditing
List all secrets in namespace:
```bash
kubectl -n <ns> get secret
```

### DO NOT COMMIT
Real `.env` files, raw passwords, API keys.

# dvc_tracking.ps1

# Step 1: Add the simple model (logistic regression) to DVC
dvc add artifacts/simple_model.pkl

# Step 2: Add the complex model (random forest) to DVC
dvc add artifacts/complex_model.pkl

# Step 3: Track the new .dvc files with Git
git add artifacts/simple_model.pkl.dvc artifacts/complex_model.pkl.dvc .gitignore

# Step 4: Commit the DVC tracking changes
git commit -m "Track simple and complex models using DVC"

# Step 5: Push DVC metadata and Git commits to the remote repo (optional, if you have remote setup)
# dvc push
# git push

Write-Output "DVC tracking completed for both models."

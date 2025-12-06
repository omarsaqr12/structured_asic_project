@echo off
cd /d "c:\Users\AUC\Downloads\dd2\structured_asic_project"

echo === Creating new branch from main ===
git fetch origin
git checkout main
git pull origin main
git checkout -b feature/def-generation

echo === Adding and committing changes ===
git add -A
git commit -m "Add DEF generation and CTS improvements"

echo === Pushing to GitHub ===
git push -u origin feature/def-generation

echo.
echo === Done! ===
echo Branch pushed: feature/def-generation
echo URL: https://github.com/omarsaqr12/structured_asic_project/tree/feature/def-generation
pause

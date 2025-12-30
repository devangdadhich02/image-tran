@echo off
echo Installing dependencies for beta-VAE project...
echo.

pip install -r requirements.txt

echo.
echo Installation complete!
echo.
echo Verifying installation...
python test_imports.py

echo.
echo If you see [SUCCESS], you're ready to go!
echo See QUICK_START.md for next steps.
pause


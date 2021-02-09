#!/bin/bash

echo "Test environment information"
echo "----------------------------"
echo "Python version: $(python --version 2>/dev/null || echo 'Python not installed')"
echo "Ray version: $(ray --version 2>/dev/null || echo 'Ray not installed')"
echo "Installed pip packages:"
echo "$(python -m pip freeze 2>/dev/null || echo 'Pip not installed')"
echo "----------------------------"

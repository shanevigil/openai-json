#!/bin/bash
# regenerate_docs.sh

# Step 1: Clean old files
rm -rf docs/source/modules.rst
rm -rf docs/source/openai_json.rst
rm -rf docs/build/*

# Step 2: Generate .rst files
sphinx-apidoc -o docs/source/ openai_json/

# Step 3: Build the documentation
make -C docs/ html

echo "Documentation regenerated successfully. Check docs/build/html/index.html"

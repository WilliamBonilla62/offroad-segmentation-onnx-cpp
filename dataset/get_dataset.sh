#!/bin/bash

# Exit if any command fails
set -e

echo "🚀 Downloading GOOSE 2D datasets in parallel..."

# Step 1: Download datasets in parallel
wget -c https://goose-dataset.de/storage/goose_2d_train.zip &
wget -c https://goose-dataset.de/storage/goose_2d_val.zip &
wget -c https://goose-dataset.de/storage/goose_2d_test.zip &
wait

echo "✅ Download complete."

# Step 2: Unzip datasets
echo "📦 Unzipping datasets..."
unzip -q goose_2d_train.zip -d gooseEx_2d_train
unzip -q goose_2d_val.zip -d gooseEx_2d_val
unzip -q goose_2d_test.zip -d gooseEx_2d_test
echo "✅ Unzip complete."

# Step 3: Create final directory structure
echo "📂 Creating final structure..."
mkdir -p goose-dataset/images/{test,train,val}
mkdir -p goose-dataset/labels/{train,val}

# Step 4: Move files into place
echo "🚚 Moving files..."
mv gooseEx_2d_test/{CHANGELOG,goose_label_mapping.csv,LICENSE} goose-dataset/
mv gooseEx_2d_test/images/test/* goose-dataset/images/test/
mv gooseEx_2d_train/images/train/* goose-dataset/images/train/
mv gooseEx_2d_train/labels/train/* goose-dataset/labels/train/
mv gooseEx_2d_val/images/val/* goose-dataset/images/val/
mv gooseEx_2d_val/labels/val/* goose-dataset/labels/val/

echo "✅ GOOSE Dataset is ready in ./goose-dataset/"

rm -rf goose_2d_*.zip gooseEx_2d_*

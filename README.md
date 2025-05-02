# 🚙 Offroad Segmentation ONNX C++ 🌲

## 🔍 Overview
This project demonstrates the complete lifecycle of a semantic segmentation model for offroad environments - from dataset preparation to training to production deployment. Its primary goal is to benchmark Python vs C++ inference performance using ONNX as the interchange format.

## 🎯 Key Features
- Complete ML pipeline demonstration (dataset → training → production)
- Performance benchmarking between Python and C++ inference
- Reproducible environment using Docker 🐳
- ONNX model format for cross-platform compatibility
- Semantic segmentation for offroad navigation

## 🏗️ Project Structure
```
├── dataset/                # Dataset handling code
│   └── goose-dataset/      # GOOSE dataset for offroad segmentation
├── docker/                 # Docker configuration
├── model/                  # Model architecture definitions
├── onnx_inference/         # C++ inference implementation
│   └── src/                # C++ source code
└── wandb/                  # Weights & Biases experiment tracking
```

## 📊 GOOSE Dataset
This project utilizes the GOOSE (Ground Offroad Outdoor SEgmentation) dataset, which provides labeled images for offroad environments. The dataset includes:
- RGB and NIR (Near Infrared) image pairs
- Semantic segmentation masks for traversability analysis
- Various offroad scenes and conditions

## 🛠️ Getting Started
```bash
# Clone the repository
git clone https://github.com/yourusername/offroad-segmentation-onnx-cpp.git
cd offroad-segmentation-onnx-cpp

# Run the docker container
docker-compose up -d

# Download the dataset
./dataset/get_dataset.sh
```

## 🧠 Model Training
```bash
# Train the segmentation model
python train_goose.py
```

## 🚀 Inference

### Python Inference
```python
# Import the model from the factory
from model.factory import create_model

# Load an image and run inference
# (See train_goose.py for complete example)
```

### C++ Inference
```bash
# Build the C++ inference engine
cd onnx_inference
mkdir build && cd build
cmake ..
make

# Run inference on an image
./segmentation_inference <image_path>
```

## 📈 Benchmarking
This project allows direct comparison between Python and C++ inference performance:
- Throughput (FPS)
- Latency (ms)
- Memory usage
- CPU/GPU utilization

## 🐳 Docker Environment
All dependencies and configurations are encapsulated in Docker, ensuring consistent and reproducible results across different machines.

```bash
# Build and start the Docker environment
docker-compose up -d
```

## 📝 License
See the LICENSE file for details.
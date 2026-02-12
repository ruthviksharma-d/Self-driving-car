# 🚗 Self-Driving Car | Behavioral Cloning Project

An end-to-end **Self-Driving Car Simulation** project built using **Python, Deep Learning, and Behavioral Cloning**.  
This project trains a neural network model to predict steering angles from driving images and runs the trained model inside a simulator for autonomous control.

---

## 👨‍💻 Contributors

This project was developed collaboratively by:

- **ruthviksharma-d**  
- **Gitterman9000**

---

## 📌 Project Overview

This project demonstrates how a self-driving car can learn steering behavior directly from human driving data.

It includes:

- Image-based training dataset  
- Driving log with steering angles  
- Deep learning model training  
- Multiple saved trained models  
- Real-time simulation using `drive.py`  

The model learns from recorded driving data (`driving_log.csv`) and predicts steering angles for new incoming frames during simulation.

---

## 🗂️ Repository Structure

```
📦 Self-driving-car
├── 📁 data/
│   ├── 📁 img/                 # Captured driving images
│   └── 📄 driving_log.csv      # Steering angle + image paths
│
├── 📁 model/
│   ├── 📄 Dodel-01.h5
│   ├── 📄 Dodel-02.h5
│   ├── 📄 Dodel-03.h5
│   ├── 📄 Dodel-m2-01.h5
│   ├── 📄 Dodel-m2-02.h5
│   └── 📄 m1-01.h5             # Trained model files
│
├── 📄 Model_Creator.ipynb      # Model training notebook
├── 📄 drive.py                 # Simulation driving script
├── 📄 requirements.txt         # Project dependencies
├── 📄 LICENSE
└── 📄 README.md
```

---

## 🧠 How It Works

### 1️⃣ Data Collection

- Images captured from center camera  
- Steering angles recorded in `driving_log.csv`  
- Stored inside `data/` directory  

---

### 2️⃣ Model Training

Training is performed in:

```
Model_Creator.ipynb
```

Steps involved:

- Load image paths and steering values  
- Preprocess images (resize, normalize, crop, etc.)  
- Build CNN model (Behavioral Cloning architecture)  
- Train on dataset  
- Save trained `.h5` model inside `model/` folder  

---

### 3️⃣ Running Autonomous Simulation

After training a model:

```bash
python drive.py model/m1-01.h5
```

The script will:

- Load selected `.h5` model  
- Connect to the driving simulator  
- Process live frames  
- Predict steering angles  
- Control the vehicle autonomously 🚘  

---

## ⚙️ Installation Guide

### Clone the Repository

```bash
git clone https://github.com/ruthviksharma-d/Self-driving-car.git
cd Self-driving-car
```

### Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🛠️ Technologies Used

- Python  
- NumPy  
- Pandas  
- OpenCV  
- TensorFlow / Keras  
- Matplotlib  
- Jupyter Notebook  

---

## 📊 Model Files

The `model/` directory contains multiple trained experiments:

- `Dodel-01.h5`  
- `Dodel-02.h5`  
- `Dodel-03.h5`  
- `Dodel-m2-01.h5`  
- `Dodel-m2-02.h5`  
- `m1-01.h5`  

Each represents a different training configuration or experiment version.

---

## 🚀 Future Improvements

- Improve dataset diversity  
- Implement lane detection module  
- Add throttle prediction  
- Integrate object detection  
- Deploy to embedded systems (Jetson Nano / Raspberry Pi)  

---

## 📄 License

This project is licensed under the MIT License.  
See the `LICENSE` file for details.

---

## 🙌 Contributing

Contributions are welcome!

1. Fork the repository  
2. Create a new branch  
3. Make your changes  
4. Commit and push  
5. Open a pull request  

---

## ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub!

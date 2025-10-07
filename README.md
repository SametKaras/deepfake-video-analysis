# **Facial Expression Analysis System**

This project is developed for the purpose of analyzing facial expressions.  
It uses techniques such as **CNN-based image analysis** to detect and classify facial expressions from video data.

---

## **📌 1. Requirements and Installation**

The project requires the following Python dependencies:

| 🔧 Component            | 🏷️ Version |
| ---------------------- | ---------- |
| **CUDA Version**       | 11.8       |
| **Python Version**     | >=3.8      |
| **PyTorch Version**    | >=1.10.0   |
| **TorchVision Version**| >=0.11.0   |
| **OpenCV Version**     | >=4.5.3    |
| **Pandas**             | >=1.3.3    |
| **Matplotlib**         | >=3.4.3    |
| **Albumentations**     | >=1.4.15   |
| **ONNXRuntime**        | >=1.10.0   |
| **tqdm**               | >=4.62.3   |

---

## **🐳 2. Using Docker**

The project is configured to run inside a Docker container.  
This ensures a standardized working environment for testing and deployment.

### **📌 2.1. Build the Docker Image**

To build the Docker image, run the following command:

```bash
sudo docker build -f docker/Dockerfile -t facial_expression_analysis .
```

This command will create a Docker image that includes all necessary dependencies based on the provided Dockerfile.

### **📌 2.2. Run the Docker Container**

To run the created image inside a Docker container:

```bash
sudo docker run --gpus all \
    -v /path/to/input/:/input \  # Path containing the input CSV and video files
    -v /path/to/output:/output \ # Path where output.csv will be saved
    facial_expression_analysis
```

Details:

- `/path/to/input/`: The directory that contains both the `input.csv` file and video files to be processed  
- `/path/to/output/`: The directory where the generated output file (`output.csv`) will be saved

> ⚠️ The `input.csv` file and video files **must be in the same input directory**!

---

## **⚠️ 3. Notes for Docker Usage**

- The output directory should be mounted as a volume inside the Docker container.
- Set appropriate permissions to access the output directory from outside Docker:

```bash
sudo chown -R $USER:$USER /path/to/output/
```

---

## **📂 4. Project Directory Structure**

```bash
📂 project_root/
│
├── 📂 src
│   ├── algorithm.py         # Main analysis logic
│   ├── model.pt             # Pre-trained CNN model
│
├── 📂 docker
│   ├── Dockerfile           # Docker environment definition
│
├── 📂 dataset
│   ├── dataset.csv          # Optional: training/label data
│   ├── video_0001.png       # Sample extracted video frames
│   ├── video_0002.png
│   └── ...
│
└── README.md
```
## **🔒 5. Dataset Privacy Notice**

The dataset used during the development of this project is private and confidential, and therefore cannot be shared publicly.
For ethical and privacy reasons, this repository does not include any real facial images or video data.

If you wish to reproduce or test the system, you must create and use your own dataset that complies with privacy and consent regulations.
Your dataset should contain labeled facial expressions (e.g., happy, sad, angry, surprised, neutral, etc.) in order to train or fine-tune the model.

⚠️ Please ensure all collected data is used only for research and non-commercial purposes, with the consent of all participants.

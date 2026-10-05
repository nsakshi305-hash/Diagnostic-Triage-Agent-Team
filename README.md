\# Diagnostic Triage Agent Team



\## AI-Based Medical Image Analysis Prototype



A modular, agent-based medical image analysis prototype that processes medical images through a structured pipeline consisting of image triage, AI-based finding classification, image segmentation, quantitative evaluation, and automated reporting.



The current implementation uses \*\*gastrointestinal endoscopy images from the Kvasir-SEG dataset\*\* for segmentation development and evaluation.



> \\\*\\\*Disclaimer:\\\*\\\* This project is developed for academic and research purposes only. It is not a clinical diagnostic system, and its outputs must not be used for medical decision-making.



\---



\## Overview



The system is designed as a multi-agent workflow in which each component performs a specific stage of the image analysis process.



```text

Input Medical Image

\&#x20;       │

\&#x20;       ▼

┌──────────────────────┐

│    Triage Agent      │

│ Image Quality Check  │

└──────────┬───────────┘

\&#x20;          │

\&#x20;          ▼

┌──────────────────────┐

│ Medical Classifier   │

│ AI Finding Prediction│

└──────────┬───────────┘

\&#x20;          │

\&#x20;          ▼

┌──────────────────────┐

│  Segmentation Agent  │

│ Target Region Mask   │

└──────────┬───────────┘

\&#x20;          │

\&#x20;          ▼

┌──────────────────────┐

│  Evaluation Agent    │

│ Dice Score and IoU   │

└──────────┬───────────┘

\&#x20;          │

\&#x20;          ▼

┌──────────────────────┐

│  Reporting Agent     │

│ Automated Summary    │

└──────────────────────┘

```



\---



\## Key Components



\### 1. Triage Agent



Performs basic validation of the uploaded image before further processing.



Responsibilities include:



\* Image loading and validation

\* Basic image quality checks

\* Image dimension extraction

\* Accept/reject decision for the processing pipeline



\### 2. Medical Classifier Agent



Performs image classification using the pretrained:



\*\*`mmuratarat/kvasir-v2-classifier`\*\*



The classifier provides the top predicted classes and their confidence scores.



Supported classes include:



\* Polyps

\* Dyed-lifted polyps

\* Dyed-resection margins

\* Esophagitis

\* Normal cecum

\* Normal pylorus

\* Normal Z-line

\* Ulcerative colitis



The classification result represents an AI-predicted image pattern and is not a medical diagnosis.



\### 3. Segmentation Agent



Performs semantic segmentation of the target region using a transfer-learning model based on:



\*\*DeepLabV3 + MobileNetV3-Large\*\*



The model was fine-tuned using the Kvasir-SEG dataset.



The segmentation output is a binary mask representing the target region.



\### 4. Evaluation Agent



Evaluates the predicted segmentation against the corresponding ground-truth mask using:



\* \*\*Dice Score\*\*

\* \*\*Intersection over Union (IoU)\*\*



These metrics measure the spatial overlap between the predicted and reference segmentation masks.



\### 5. Reporting Agent



Generates a structured summary containing:



\* Triage status

\* Image dimensions

\* Segmentation metrics

\* Segmentation quality

\* Research/educational disclaimer



\---



\## Model



\### Segmentation Architecture



```text

DeepLabV3

\&#x20;   │

\&#x20;   └── MobileNetV3-Large Backbone

\&#x20;             │

\&#x20;             ▼

\&#x20;      Binary Segmentation

```



The trained checkpoint is stored at:



```text

outputs/trained\\\_segmentation/deeplabv3\\\_mobilenetv3\\\_kvasir.pth

```



\### Classification Model



```text

mmuratarat/kvasir-v2-classifier

```



The classifier is used to generate AI-based image finding predictions for gastrointestinal endoscopy images.



\---



\## Evaluation Metrics



\### Dice Score



Dice Score measures the overlap between the predicted segmentation and the ground-truth mask.



```text

Dice = 2 × |Prediction ∩ Ground Truth|

\&#x20;      ───────────────────────────────

\&#x20;      |Prediction| + |Ground Truth|

```



\### Intersection over Union



IoU measures the intersection of the prediction and ground truth relative to their union.



```text

IoU = |Prediction ∩ Ground Truth|

\&#x20;     ───────────────────────────

\&#x20;     |Prediction ∪ Ground Truth|

```



Higher values indicate greater overlap between the predicted and reference masks.



\---



\## Example Pipeline Output



```text

DIAGNOSTIC TRIAGE AGENT TEAM



\\\[1] TRIAGE AGENT

Status: accepted

Image Size: 622 x 529



\\\[2] SEGMENTATION AGENT

Segmentation completed successfully.



\\\[3] EVALUATION AGENT

Dice Score : 0.6700

IoU Score  : 0.5038



\\\[4] REPORTING AGENT

Segmentation Quality: Moderate



Pipeline completed successfully.

```



The displayed metrics are dependent on the input image and model prediction.



\---



\## Streamlit Interface



The project includes an interactive Streamlit dashboard for running the pipeline and visualizing results.



The interface provides:



\* Input image preview

\* AI-predicted finding

\* Top classification predictions

\* Predicted segmentation mask

\* Segmentation overlay

\* Dice Score

\* IoU Score

\* Image dimensions

\* Automated report

\* Research and educational disclaimer



\---



\## Technology Stack



| Technology   | Purpose                    |

| ------------ | -------------------------- |

| Python       | Application development    |

| PyTorch      | Deep learning              |

| Torchvision  | Segmentation architecture  |

| Transformers | Image classification       |

| OpenCV       | Image processing           |

| NumPy        | Numerical computation      |

| Pandas       | Data processing            |

| Scikit-learn | Machine learning utilities |

| Streamlit    | Web interface              |

| Git          | Version control            |

| GitHub       | Repository hosting         |



\---



\## Dataset



\### Kvasir-SEG



The segmentation component was developed using the \*\*Kvasir-SEG\*\* dataset.



The dataset contains:



\* 1,000 gastrointestinal endoscopy images

\* 1,000 corresponding segmentation masks

\* Pixel-level annotations for polyp regions



The dataset itself is \*\*not included in this repository\*\*.



\---



\## Project Structure



```text

Diagnostic-Triage-Agent-Team/

│

├── agents/

│   ├── medical\\\_classifier\\\_agent.py

│   ├── segmentation\\\_agent.py

│   ├── triage\\\_agent.py

│   ├── evaluation\\\_agent.py

│   └── reporting\\\_agent.py

│

├── outputs/

│   └── trained\\\_segmentation/

│       └── deeplabv3\\\_mobilenetv3\\\_kvasir.pth

│

├── app.py

├── main.py

├── segmentation\\\_agent\\\_trained.py

├── train\\\_segmentation.py

├── requirements.txt

├── .gitignore

└── README.md

```



\---



\## Installation



\### Clone the repository



```bash

git clone https://github.com/nsakshi305-hash/Diagnostic-Triage-Agent-Team.git

cd Diagnostic-Triage-Agent-Team

```



\### Create a virtual environment



```bash

python -m venv venv

```



\### Activate the environment



\*\*Windows:\*\*



```bash

venv\\\\Scripts\\\\activate

```



\*\*Linux/macOS:\*\*



```bash

source venv/bin/activate

```



\### Install dependencies



```bash

pip install -r requirements.txt

```



\---



\## Usage



\### Run the pipeline



```bash

python main.py

```



\### Launch the Streamlit application



```bash

streamlit run app.py

```



The Streamlit interface can then be accessed through the local URL displayed in the terminal.



\---



\## Training



The segmentation training workflow is provided in:



```text

train\\\_segmentation.py

```



The trained model uses a two-class segmentation setup:



```text

Class 0 — Background

Class 1 — Target Region

```



The resulting model checkpoint is stored in:



```text

outputs/trained\\\_segmentation/

```



\---



\## Project Objectives



The project aims to demonstrate:



1\. A modular multi-agent architecture for medical image analysis.

2\. Automated image-quality triage.

3\. AI-based image finding classification.

4\. Deep-learning-based image segmentation.

5\. Quantitative segmentation evaluation.

6\. Automated result reporting.

7\. Integration of the complete workflow into an interactive application.



\---



\## Future Work



Potential improvements include:



\* Training with larger and more diverse datasets

\* Improving segmentation accuracy

\* Out-of-domain image detection

\* Confidence calibration

\* Explainable AI techniques

\* Additional medical imaging modalities

\* Model benchmarking and comparison

\* GPU-based training and inference optimization

\* Improved reporting and visualization



\---



\## Disclaimer



This software is an \*\*academic and research prototype\*\*.



It is not intended to diagnose, treat, or prevent any medical condition. The predictions, segmentation results, and generated reports should not be interpreted as medical advice or clinical diagnosis.



\---



\## Author



***Sakshi Narayan***



B.Tech CSE — AI/ML

Centurion University of Technology and Management, Bhubaneswar



\*\*Areas of Interest:\*\* Artificial Intelligence, Machine Learning, Computer Vision and Data Analytics.


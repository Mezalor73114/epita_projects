# 📂 EPITA Data Science & AI Projects
## 👤 About
I am a final-year Computer Engineering Student at EPITA, majoring in Data Science and Artificial Intelligence (SCIA).

This repository serves as a technical portfolio gathering various academic projects completed during my coursework. Most projects are implemented in Jupyter Notebooks and are accompanied by PDF reports detailing the theoretical approach and analysis results.

## 🚀 Projects Overview

| Project Directory | Description | Tech Stack |
| :---------------- |:-----------:| ----------:|
| 📁 Medical_Image_Seg | 3D Computer Vision: Segmentation of hepatic vessels and tumors in CT-scans. Comparative study between Classical (ITK) and Deep Learning (MONAI) approaches. | MONAI, PyTorch, ITK, 3D Data |
| 📁 Diffusion | Generative AI: Implementation of a Denoising Diffusion Probabilistic Model (DDPM) to generate video game sprites. Includes trained weights and theoretical report. | PyTorch, Generative Models |
| 📁 US_Accidents | End-to-End ML Pipeline: Comprehensive analysis of US traffic accidents. Tasks include missing value imputation, severity prediction, and distance prediction. | Scikit-learn, Pandas, Data Cleaning |
| 📁 Snake_Reinforcement_Learning | Reinforcement Learning: Implementation of an agent learning to play Snake using Reinforcement Learning techniques. | RL, Q-Learning |
| 📁 Conformal_Prediction | Uncertainty Quantification: Application of conformal prediction methods on hospital data for regression (insurance costs) and classification (cardio diagnosis). | Conformal Prediction, Statistics |
| 📁 DataViz | Exploratory Analysis: Visualization project analyzing Global YouTube Statistics to extract trends and insights. | Matplotlib, Seaborn |
| 📁 EDA | Automated Analysis Tool: A Python application (app.py) designed to perform automated Exploratory Data Analysis on datasets. | Python Scripting, App Dev |

## 🛠️ Usage
1. Clone the repository:
```
git clone [https://github.com/Mezalor73114/epita_projects.git](https://github.com/Mezalor73114/epita_projects.git)
cd epita_projects
```
2. Dependencies:
  - Most projects rely on standard Data Science libraries (numpy, pandas, scikit-learn, matplotlib, pytorch).
  - The Medical Imaging project requires specific libraries:
    ```
    pip install monai itk nibabel
    ```
  - The EDA project has specific requirements:
    ```
    pip install -r EDA/requirements.txt
    ```
3. Run:
  - Launch Jupyter Lab or Notebook to view .ipynb files:
    ```
    jupyter notebook
    ```
  - For the EDA app:
    ```
    cd EDA/code
    python app.py
    ```

## 📫 Contact
- Clément Martin
- 📧 clement.martin@epita.fr
- 💼 LinkedIn Profile : https://linkedin.com/in/clement-martin73114

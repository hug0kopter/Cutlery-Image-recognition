# 🥄 Cutlery Image Recognition

An end-to-end machine learning pipeline for detecting and classifying cutlery (**fork**, **spoon**, **knife**, and **ignore**) using **OpenCV** and **scikit-learn**.  
All paths are **repo-relative**, so the project runs correctly from any directory after cloning.

This repository includes the **processed dataset** (`artifacts_shared/cutlery_dataset.npz` and `dataset_meta.json`) needed for model training and deployment.  
The **original images** are hosted separately on Google Drive due to GitHub file size limits.

---

## 🚀 Quick Start

### 1️⃣ Clone and Setup
```
git clone https://github.com/<your-username>/Cutlery_Image_Recognition.git
cd Cutlery_Image_Recognition
python -m venv venv
venv\Scripts\activate        # On Windows
# or
source venv/bin/activate       # On macOS/Linux
pip install -r requirements.txt
```

### 2️⃣ Run the Model Deployment
Start the live camera recognition using your trained model:
```
python Scripts/deploy/Model_deploy_git.py
```
Press `q` or `Esc` to quit the live video.

### 3️⃣ (Optional) Retrain or Rebuild the Dataset
If you want to rebuild or retrain:

**Rebuild dataset from processed images:**
```
python Scripts/Data_aquaire_proccessing/explore_and_save_fixed.py
```

**Train models:**
```
python Scripts/Classification/train_from_saved_knn.py
python Scripts/Classification/train_from_saved_svm.py
python Scripts/Classification/train_from_saved_tree.py
```

---

## 📦 Repository Structure

```
Cutlery_Image_Recognition/
├── Scripts/
│   ├── Data_aquaire_proccessing/
│   │   ├── acquire_cutlery_img_git.py
│   │   ├── Segment_data_Path_git.py
│   │   └── explore_and_save_fixed.py
│   ├── Classification/
│   │   ├── train_from_saved_knn.py
│   │   ├── train_from_saved_svm.py
│   │   └── train_from_saved_tree.py
│   └── deploy/
│       └── Model_deploy_git.py
│
├── artifacts_shared/
│   ├── cutlery_dataset.npz
│   ├── dataset_meta.json
│   └── plots/
│
├── data_and_features/
│   ├── .gitkeep
│   ├── Data_original/
│   │   └── .gitkeep
│   └── Data_proccesed/
│       └── .gitkeep
│
├── requirements.txt
└── README.md
```

---

## 📸 Original Dataset (Google Drive)

Due to GitHub size limits, the **original image dataset** is available as a ZIP file on Google Drive.

**Download here:**  
[Google Drive – Cutlery Raw Dataset (ZIP)](PASTE_YOUR_GOOGLE_DRIVE_LINK_HERE)

After downloading, extract the ZIP file into:
```
Cutlery_Image_Recognition/data_and_features/
```

So that it looks like:
```
Cutlery_Image_Recognition/
└── data_and_features/
    └── Data_original/
        ├── fork/
        ├── spoon/
        ├── knife/
        └── ignore/
```

---

## 🧰 Empty Folder Placeholders (.gitkeep)

Git does not track empty folders.  
Small `.gitkeep` files are included so the folder structure remains visible after cloning the repo.

---

## ⚙️ Troubleshooting

- **Camera not detected**  
  Change the camera source number (`CAM_SRC`) in `Model_deploy_git.py` to 0 or 1.  
  Close other programs that might be using the camera.  
  If video is slow, reduce the frame size (for example `(1280, 720)`).

- **Model or dataset not found**  
  Ensure the files `cutlery_knn.joblib` and `dataset_meta.json` exist in the `artifacts_shared` folder.  
  If missing, rebuild the dataset using the data processing scripts and retrain the model.

- **Modules missing (e.g. cv2, numpy)**  
  Run `pip install -r requirements.txt`.  
  Make sure you are using the correct Python environment.

- **Unstable or incorrect predictions**  
  The model is sensitive to lighting.  
  Use a **dark or black background** for best results.  
  Avoid shiny or reflective surfaces.

- **Paths not working**  
  Always run scripts from inside the main project folder.  
  The project uses relative paths, so folder names must remain the same.

---

## 🧪 Experimental Setup

The model was trained using **wooden cutlery** available at some **Albert Heijn** supermarkets in the Netherlands.  
At certain locations, this cutlery is **free to take** when purchasing other products.

All photos were captured using a **3D-printed rotating fixture** powered by a **5 V DC motor** with a built-in reduction gearbox.  
The fixture rotated the cutlery slowly, allowing photos from many angles.  
The holder caused only **minor visual obstruction**, with no measurable effect on recognition accuracy.

The segmentation and model were optimized for **non-reflective, bright materials placed on a dark or black background**.  
Reflective or metallic cutlery will not work properly with this version.

---

## 🧩 Requirements

Install dependencies with:
```
pip install -r requirements.txt
```

Contents of `requirements.txt`:
```
numpy
opencv-python
imutils
joblib
scikit-learn
matplotlib
seaborn
```

---

## 📄 License

**MIT License**  
You may use, modify, and distribute this project freely with attribution.

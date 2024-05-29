Here is the  `README.md` file with the provided information structured clearly:


Indian-Currency-Classification

An application for classifying new Indian currency notes using the CNN algorithm, a deep learning technique. It is implemented with the help of **OpenCV** and **Scikit-learn** libraries. The dataset used for training consists of **1050** images of new currency notes (INR). It includes denominations of **10, 20, 50, 100, 200, 500, and 2000** rupees notes (i.e., 7 classes with 150 images each).

Dataset link: [Indian Currency Dataset](https://www.kaggle.com/datasets/jayaprakashpondy/indian-currency-dataset)

## Methodology

In this project, we utilize a combination of global and local features for image classification and leverage a Convolutional Neural Network (CNN) for training and inference. Below is an overview of the methodology:

### Feature Extraction

**Global Features:**
- **Hu Moments (Shape):** Extract shape-based features using Hu Moments, which are invariant to image transformations such as translation, scaling, and rotation.
- **Haralick Features (Texture):** Extract texture features using Haralick descriptors, which capture the spatial distribution of gray levels in an image.
- **Colour Histogram (Colour):** Compute colour histograms to capture the distribution of colours within the image.

**Local Features:**
- **Bag of Visual Words (BoVW) with SIFT:**
  - Use the Scale-Invariant Feature Transform (SIFT) to detect and describe local features.
  - Build a visual vocabulary using the BoVW approach, clustering the SIFT descriptors to create a codebook.

### Model Training

We train a Convolutional Neural Network (CNN) on the extracted features to classify the images. The CNN is designed to learn complex patterns and relationships from the input features.

### Inference

1. **Preprocessing:** Preprocess the input image to ensure consistency with the training data.
2. **ROI Extraction:** Extract rectangular Regions of Interest (ROIs) from the preprocessed image using OpenCV. This involves detecting and cropping rotated contours to focus on relevant parts of the image.
3. **Feature Extraction from ROIs:** Extract the same global and local features from the ROIs as during the training phase.
4. **Classification:** Feed the extracted features from the ROIs into the trained CNN model to predict the class labels.

## Dependencies

- Python 3
- Scikit-learn
- OpenCV with contrib modules
- Mahotas
- Pickle
- Joblib

You can install these dependencies using pip:

```bash
pip install scikit-learn opencv-python opencv-contrib-python mahotas pickle-mixin joblib
```

## Demo

### How to run

1. **Feature Extraction:**

   ```bash
   python bovw.py
   ```

2. **Hyperparameter Tuning:**

   ```bash
   python hyper_train.py
   ```

3. **Model Training:**

   ```bash
   python train.py
   ```

4. **Model Inference:**

   ```bash
   python predict.py
   ```

5. **Currency Classification:**

   ```bash
   python currency.py
   ```

### Training Results

**Best Parameters:**

```plaintext
bootstrap=True, criterion='gini',
min_samples_leaf=1, min_samples_split=2,
min_weight_fraction_leaf=0.0, n_estimators=200,
n_jobs=-1,verbose=1
```

**Cross Validation Accuracy:** 0.98 (+/- 0.01)

**Confusion Matrix:**

|      | 10 | 20 | 50 | 100 | 200 | 500 | 2000 |
|------|----|----|----|-----|-----|-----|------|
| **10**   | 28 | 0  | 1  | 0   | 2   | 0   | 0    |
| **20**   | 0  | 24 | 0  | 0   | 0   | 1   | 0    |
| **50**   | 0  | 0  | 27 | 0   | 0   | 0   | 0    |
| **100**  | 0  | 0  | 0  | 32  | 0   | 0   | 0    |
| **200**  | 2  | 0  | 0  | 0   | 31  | 0   | 0    |
| **500**  | 0  | 1  | 0  | 0   | 0   | 26  | 0    |
| **2000** | 0  | 0  | 0  | 0   | 0   | 0   | 35   |

### Sample Output

Home page.png
Real Currency Detection.png
Fake Currency Detection.png



## Authors

- Sai Madhuri K
- Jyothi L
- Akhil K
- Guna Koushal N

## Acknowledgments

- "https://kushalvyas.github.io/BOV.html"
- "https://gogul.dev/software/image-classification-python"
- "https://github.com/briansrls/SIFTBOW"
- "https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/"
- "https://towardsdatascience.com/optimizing-hyperparameters-in-random-forest-classification-ec7741f9d3f6"
- "https://stackoverflow.com/questions/11627362/how-to-straighten-a-rotated-rectangle-area-of-an-image-using-opencv-in-python/48553593#48553593"
- "https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html" better understanding for user

```

This `README.md` file now provides a comprehensive overview of your project, including the project description, methodology, dependencies, demo instructions, training results, sample output, versioning, authorship, and acknowledgments.

# AI vs Human Art — Binary Image Classification (TensorFlow/Keras CNN)

This project trains a convolutional neural network (CNN) to classify images into two classes:
**AI-generated images** vs **human-created art**.

The dataset started from a Kaggle source and was expanded with additional examples to strengthen both classes.

## Dataset
Directory structure:
Data/
  AI_Created/
  Human_Created/


Data is already split into two classes at the folder level.

**Train/Validation/Test split:** 70% / 10% / 20%  
(Implemented by splitting dataset batches after loading.)

**Current test set (latest split):**
- AI Image: 419 images  
- Human Art: 408 images  

## Preprocessing & Cleaning
The pipeline follows the notebook:
- Remove images that are not `jpeg/jpg/png`
- Load images using `tf.keras.utils.image_dataset_from_directory`
- Normalize pixel values to `[0,1]` by dividing by the maximum pixel value observed in a batch
- Default image size used by the loader is **256×256**, matching the model input shape

## Model (CNN from scratch)
TensorFlow/Keras Sequential model:
- Conv2D(16, 3×3, ReLU) + MaxPool
- Conv2D(32, 3×3, ReLU) + MaxPool
- Conv2D(16, 3×3, ReLU) + MaxPool
- Flatten
- Dense(256, ReLU)
- Dense(1, Sigmoid)

**Compile:**
- Optimizer: Adam
- Loss: Binary Crossentropy
- Metric: Accuracy

**Training:**
- 10 epochs
- TensorBoard logging enabled

## Results
This repo includes an evaluation script that reports:
- Accuracy
- Precision
- Recall
- F1
- Confusion matrix (saved as an image)

> Metrics may vary depending on the exact dataset contents and split ordering.

## Run it end-to-end

### 1) Setup
```bash
git clone https://github.com/rstrauss127/img_classification.git
cd img_classification

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```
### 2) Ensure data is in Data/
Data/AI_Created/
Data/Human_Created/ 
Kaggle Data Source: https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images

### 3) Train
python src/train.py --data_dir Data --epochs 10 --batch_size 32 --seed 123

Outputs:
  outputs/model.keras
  outputs/train_history.json
  outputs/loss.png
  outputs/accuracy.png

### 4) Evaluate
python src/eval.py --data_dir Data --model_path outputs/model.keras --seed 123

## Notes / Limitations

This project presented as a CNN classification case study: data cleaning, preprocessing, training, and evaluation.The classification algorithm is sensitive to:
  Generator drift (new AI models/styles)
  Dataset source bias
  Label noise from web-sourced images


Task_3_Skillcraft
Task 03: Support Vector Machine (SVM) Implementation
Machine Learning Internship at SkillCraft Technology

I’m excited to share that I’ve successfully completed Task 03 of my Machine Learning internship at SkillCraft Technology, which involved building an SVM classifier to distinguish between images of cats and dogs using a dataset from Kaggle.

📌 Task Objective
To develop an image classification model using Support Vector Machine (SVM) to accurately identify whether an input image is of a cat or a dog.

🛠 Approach and Methodology
Used a dataset containing 5,000 labeled images for training and 12,500 images for testing

Resized all images to 64×64 pixels and converted them into feature vectors

Split the dataset into training and validation sets for evaluation

Trained an SVM classifier using the scikit-learn library with tuned hyperparameters

Evaluated the model using precision, recall, and F1-score metrics

Saved the trained model and generated predictions for test data

Exported predictions and confidence scores into a structured CSV file for further analysis

✅ Key Outcomes
Achieved a validation accuracy of 73.18%, with well-balanced performance across both classes

Classified 12,500 unseen test images, including confidence scores for each prediction

Successfully implemented a classical machine learning solution for image classification without deep learning

💡 Skills and Concepts Strengthened
Applied image preprocessing techniques for traditional ML models

Gained practical experience with Support Vector Machine (SVM) for binary classification

Learned to evaluate models using validation reports and metrics

Implemented model serialization and automated batch prediction generation

Managed and analyzed results using CSV export and confidence scoring

💻 Output Snapshot:
bash
Copy
Edit
PS C:\codes\internship_task> python -u "c:\codes\internship_task\task3.py"
Initialized classifier with image size: (64, 64)
Training directory: C:/Users/Yatin Singh/Downloads/dogs-vs-cats/train/train
Test directory: C:/Users/Yatin Singh/Downloads/dogs-vs-cats/test1/test1
Loading training data...
Processing training images: 100%|█████████████████████████████████████████████████████████████████| 25000/25000 [05:47<00:00, 72.03it/s]
Loaded 25000 training samples
Feature vector size: 1086
Class distribution: [12500 12500]
Training SVM classifier...
Preprocessing features...
Validation Accuracy: 0.7318

Validation Classification Report:
              precision    recall  f1-score   support

         Cat       0.73      0.74      0.73      2500
         Dog       0.74      0.72      0.73      2500

    accuracy                           0.73      5000
   macro avg       0.73      0.73      0.73      5000
weighted avg       0.73      0.73      0.73      5000

Model saved to svm_cat_dog_model.pkl
Making predictions on test images...
Processing test images: 100%|█████████████████████████████████████████████████████████████████████| 12500/12500 [04:42<00:00, 44.25it/s]
Predictions saved to test_predictions.csv

Prediction completed! Made predictions for 12500 test images.
First 5 predictions:
1.jpg: dog (confidence: 0.725)
10.jpg: dog (confidence: 0.714)
100.jpg: cat (confidence: 0.711)
1000.jpg: dog (confidence: 0.809)
10000.jpg: dog (confidence: 0.879)
PS C:\codes\internship_task>
Looking forward to diving deeper into more advanced machine learning techniques in the upcoming tasks and continuing my journey in applied AI and data science.

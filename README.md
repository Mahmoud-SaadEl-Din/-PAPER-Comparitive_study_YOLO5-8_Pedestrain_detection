# Pedestrian Detection Using YOLO Algorithm: An Experimental Study

This repository summarize my work in my paper titled **"Pedestrian Detection Using YOLO Algorithm: An Experimental Study."** The paper explores various YOLO versions, such as YOLOv5, YOLOv6, YOLOv7, and YOLOv8, and evaluates their effectiveness in real-time object detection (pedestrain detection) tasks.

# Power of YOLO
Unlike traditional methods that require multiple passes over an image, YOLO is a real-time object detection system that predicts bounding boxes and class probabilities in a single pass. It revolutionized the field by achieving impressive speed and accuracy trade-offs.

In YOLO, the input image is divided into a grid, and each grid cell is responsible for detecting objects present in its spatial region. Multiple bounding box predictions are made within each grid cell, along with corresponding class probabilities. These predictions are refined using convolutional neural networks (CNNs) trained on large-scale datasets.

# A Quick look over YOLO5:8
The general YOLO model is made up of a convolutional neural network (CNN) backbone, a neck, and a head.

### YOLOv5
- **Backbone:** YOLOv5 utilizes CSP-Darknet53, an optimized version of Darknet53
  - **Key Features:** Cross Stage Partial Network (CSP), Spatial Pyramid Pooling Fast (SPPF), SiLU (Swish) Activation
- **Neck:** YOLOv5 incorporates a modified Path Aggregation Network (CSP-PAN)
- **Head:** The final layer, adapted from YOLOv3
- **Loss Functions:** Combination of Localization Loss & Confidence Loss & Classification Loss

### YOLOv6
- **Backbone:** **RepBlock:** Lightweight and efficient for smaller models. & **CSP-StackRep Block:** More powerful backbone for larger models, capturing intricate visual patterns.
- **Neck:** Modified PAN (Path Aggregation Network) topology, enhancing multi-scale feature fusion.
- **Head:** Efficient decoupled head with a hybrid-channel strategy, optimizing classification and regression tasks separately.
- **Loss Function:** Combination of Classification Loss (VariFocal Loss) & Box Regression Loss & Knowledge Distillation (Self-distillation using KL divergence) & Knowledge Distillation
- **Special Features:** Reparameterization and Quantization (Techniques for optimizing model deployment and inference efficiency) & Knowledge Distillation (Improves accuracy by aligning the predictions of smaller models with those of larger, pre-trained models)

### YOLOv7

- **Backbone:** **E-ELAN Computational Block:** Expands input feature map channels, shuffles feature order, and merges features from different groups, enhancing learning ability without disrupting the gradient path.
  
- **Neck:** **CSPSPP+(ELAN, E-ELAN)PAN:** A combination of Spatial Pyramid Pooling (CSPSPP) and Efficient Layer Aggregation Network (ELAN) blocks. This architecture pools features of different scales to improve accuracy and efficiency.
- **Head:** A series of convolutional layers with batch normalization and ReLU activation functions, concluding with a softmax layer for class prediction. This head architecture ensures efficient and accurate prediction of bounding boxes and object classes.
- **Loss Functions:** Combining Loss Functions, a weighted combination of loss functions (Binary Cross-Entropy Loss & Categorical Cross-Entropy Loss & Smooth L1 Loss) is employed, optimizing the importance of each loss type to enhance bounding box and label prediction accuracy.

### YOLOv8
- **Backbone:** Utilizes CSPDarknet53 as a feature extractor, similar to YOLOv5 but replaces the CSPLayer with the C2f module. The C2f module is a cross-stage partial bottleneck with two convolutions, enhancing feature extraction.
- **Neck:** The C2f module acts as a more efficient and accurate bottleneck layer that combines high-level features with contextual information, improving detection accuracy by integrating object details with surrounding context.
- **Head:** Comprises a pair of segmentation heads followed by detection heads similar to YOLOv8. The segmentation heads are responsible for semantic segmentation, while the detection heads forecast object bounding boxes, classes, and confidence scores. The final prediction layer combines these outputs for object detection.
- **Loss Functions:** Combination of CIoU (Complete IoU, a modified version of IoU loss) & DFL (Decoupled Fully-connected with Label Smoothing) Loss & Binary Cross-Entropy Loss
- **Special Features:** **Efficient Bottleneck Layers (C2f):** Enhances feature extraction by combining high-level object details with contextual information, boosting detection performance. & **Segmentation and Detection Heads:** Dual-purpose heads improve both segmentation and object detection, increasing overall model versatility and accuracy.

### Comparison summarization
![Alt text](/summary_table.png)

# Related Work

This section reviews existing literature on pedestrian detection and the use of YOLO in various applications. It provides a background on previous studies and situates the current research within the broader context of object detection in computer vision.

## Methodology

### Dataset Description

The **WiderPerson** dataset is utilized in this study to evaluate the performance of different YOLO versions for pedestrian detection. The dataset is known for its highly diverse and dense annotations, providing a comprehensive benchmark for pedestrian detection tasks.

- **Diversity and Scale**: The WiderPerson dataset contains a significantly higher number of annotations compared to previous datasets like CityPersons. It offers more than ten times the number of bounding boxes and covers a wider range of scales. The distribution of pedestrian sizes in the dataset is relatively uniform, making it a robust choice for training and testing.

- **Density**: On average, there are approximately 28.87 persons per image, which is considerably higher than what is found in other pedestrian detection datasets. This high density ensures that models trained on this dataset are well-equipped to handle crowded scenes.

- **Annotations and Labels**: The dataset provides five fine-grained labels:
  - Pedestrians
  - Riders
  - Partially-visible persons
  - Crowd
  - Ignore regions

  These labels help in distinguishing between different types of objects and scenarios, such as partially visible pedestrians or groups of people.

- **Quality Assurance**: Multiple annotators have carefully checked and pre-filtered the annotations to ensure their quality and reliability. The dataset is divided into three subsets:
  - Training set
  - Validation set
  - Testing set

  This facilitates comprehensive evaluation and model benchmarking.

- **Benchmarking**: An online benchmark is also available for the WiderPerson dataset, allowing researchers to compare their models' performance with existing methods.

The WiderPerson dataset’s diversity and density make it a valuable resource for training and testing pedestrian detection models, offering a more challenging environment compared to traditional datasets.

### YOLO Algorithm Overview

A brief overview of the YOLO algorithm, explaining its architecture and how it has evolved from YOLOv1 to YOLOv8. The section also discusses the key features of each YOLO version that are relevant to pedestrian detection.

### Experimental Setup

This section describes the experimental setup, including the hardware and software used, the implementation details of each YOLO version, and the evaluation metrics applied in the study.

## Results and Discussion

### Performance Metrics

An analysis of the performance metrics for each YOLO version, such as accuracy, precision, recall, and inference time. The section compares these metrics across different versions to highlight their strengths and weaknesses.

### Comparison of YOLO Versions

A detailed comparison of YOLOv5, YOLOv6, YOLOv7, and YOLOv8, discussing how each version performs in pedestrian detection tasks and which version offers the best balance between accuracy and speed.

## Conclusion

The conclusion summarizes the key findings of the study, emphasizing the practical implications of the results for pedestrian detection in real-world applications. It also suggests potential directions for future research.

## References

A list of references cited in the paper, following the appropriate citation style.

# Solving face recognition task using Pytorch, MTCNN and ResNet18
This project uses MTCNN (https://github.com/ipazc/mtcnn) as an aligner and ResNet18 as a recognizer.
This model achieved 87.5% accuracy on validation data and 92.1% on test data.
Used dataset: https://www.kaggle.com/datasets/jessicali9530/lfw-dataset/data

```

Project's structure:
├── README.md
├── requirements.txt
├── dataset <- train and validation dataset
├── test <- test dataset
├── Face Recognition
│   ├── main.py         <- Main file of the project that launches all computations
│   │
│   ├── aligner.py      <- Aligner - a program detecting and cutting faces from the initial photo
│   │
│   └── recognizer.py   <- Recognizer - a program recognizing a person in the image that recognizer gets from the aligner
└── .gitignore

```

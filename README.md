# Solving face recognition task using Pytorch, MTCNN (https://github.com/ipazc/mtcnn) as an aligner and resnet18 as a recognizer (classifier). 
Used dataset: https://www.kaggle.com/datasets/jessicali9530/lfw-dataset/data

```

Project's structure:
├── README.md
├── requirements.txt
├── dataset <- train and validation dataset
├── test <- test dataset
├── Face Recognition
│   ├── main.py    <- Main file of the project that launches all computations
│   │
│   ├── aligner.py           <- Aligner - a program detecting and cutting faces from the initial photo
│   │
│   └── recognizer.py       <- Recognizer - a program recognizing a person in the image that recognizer gets from aligner
└── .gitignore

```
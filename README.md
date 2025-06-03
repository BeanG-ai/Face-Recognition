# Face-Recognition
Face recognition on Orin nano

## Project Structure

```
face-recognition/
│
├── data/
│   ├── known_faces/       # Database 
│   │   ├── person1/
│   │   ├── person2/
│   │   └── ...
│   ├── models/            # Pre-trained models
│   │   ├── detection/     # Face detection models
│   │   └── recognition/   # Face recognition models
│   └── test_images/       # Test images for validation
│
├── src/
│   ├── detection/         # Face detection algorithms
│   │   ├── __init__.py
│   │   └── detector.py
│   ├── recognition/       # Face recognition/identification
│   │   ├── __init__.py
│   │   └── recognizer.py
│   ├── utils/             # Helper functions
│   │   ├── __init__.py
│   │   ├── image_utils.py
│   │   └── db_utils.py
│   └── main.py        # Main pipeline implementation
│
├── tools/                 # Utility scripts
│   ├── register_face.py   # Tool to add new faces to database
│   ├── optimize_model.py  # TensorRT optimization for Jetson
│   └── benchmark.py       # Performance testing
│
├── configs/               # Configuration files
│   └── settings.json
│
├── notebooks/             # Jupyter notebooks for experimentation
│
├── tests/                 # Unit tests
│
├── requirements.txt       # Dependencies
└── README.md              # Documentation
```




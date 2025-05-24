# MRI-Neurodegenerative-Disease-Detection
<div align="center">
  <p>
    <a href="https://kajabi-storefronts-production.kajabi-cdn.com/kajabi-storefronts-production/file-uploads/blogs/22606/images/f8d6362-3e5e-c73-a7a4-e54525b5431a_banner-yolov8.png" target="_blank">
      <img width="100%" src="https://kajabi-storefronts-production.kajabi-cdn.com/kajabi-storefronts-production/file-uploads/blogs/22606/images/f8d6362-3e5e-c73-a7a4-e54525b5431a_banner-yolov8.png" alt="YOLO Vision banner"></a>
  </p>
</div>

<div align="center">
  <p>
    <a href="https://github.com/rorro6787/rorro6787/blob/main/aaaaaaa.png" target="_blank">
      <img width="100%" src="https://github.com/rorro6787/rorro6787/blob/main/aaaaaaa.png" alt="YOLO Vision banner"></a>
  </p>
</div>

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/rorro6787/neurodegenerative-disease-detector)

The Final Degree Project (TFG) will consist of the design and development of an advanced system for medical image analysis, focused on detecting brain pathologies in patients using magnetic resonance imaging (MRI) scans. The main objective of the system is to provide healthcare professionals with an effective tool to accurately and quickly identify and classify brain lesions.

An algorithm will be implemented using the YOLOv8 model, trained with the dataset defined at this link:  
[Link to the Dataset definition.](https://iplab.dmi.unict.it/mfs/ms-les-seg/#home)

Each plane of a set of MRI images will be analyzed. This algorithm will be specifically trained to work with 2D images extracted from three-dimensional (3D) representations. Its goal is to automatically detect lesions in these images. Once the lesions are identified, the algorithm will generate 2D bounding boxes that represent the intersections between the 3D images and the analyzed planes. These boxes will help visualize the exact location of the lesions in the MRI scans.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Docker Deployment](#docker-deployment) 
- [Contributing](#contributing)

## Features

## Requirements

- Python 3.X.X
- npm 
- pnpm 
- node 

## Installation

1. Clone the repository:
   
    ```sh
    git clone https://github.com/yourusername/repository_name.git
    ```

2. Navigate to the project directory:
   
    ```sh
    cd repository_name
    ```

3. (Optional) Create a virtual environment:

    ```sh
    python3 -m venv venv
    .\venv\Scripts\activate  # On macOS/Linux use 'source venv/bin/activate'
    ```

4. Select venv as your Python interpreter (in VSC):

    ```sh
    > Python: Select Interpreter
    .\venv\Scripts\python.exe  # On macOS/Linux use './venv/bin/python'
    ```

5. Install the required packages:
   
    ```sh
    pip install -r requirements.txt
    ```

6. If you add more dependencies, update the requirements file using:

    ```sh
    pip freeze > requirements.txt
    ```

## Usage

To use the system for generating mail analysis, follow these instructions:

3. **Run the Application (1)**: To use the application, first run the app.py script that hosts the service in your localhost (in case you want to use the virtual environment):

     ```sh
     python app.py
     ```

4. **Run the Application (2)**: To use the application with Docker, proceed to the next section of the README, where everything is explained in detail.
  
## Docker Deployment
If you have Docker installed on your system and prefer not to install all the dependencies and the specific Python version required to test this app, you can use Docker to run the app in a lightweight virtual environment. The project includes a Dockerfile that processes all the necessary requirements and creates a Docker image of a virtual machine that fulfills all the dependencies and runs the project perfectly. Simply run the following commands:
```sh
docker build -t mi_proyecto:latest .
docker run -it -p 5000:5000 mi_proyecto
```

## Contributors

- [![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/rorro6787) [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/emilio-rodrigo-carreira-villalta-2a62aa250/) **Emilio Rodrigo Carreira Villalta**

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

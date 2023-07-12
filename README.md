# AutoML System

This repository contains a simple AutoML system developed using Python and scikit-learn. The system is designed to automatically pre-process data, select the best model from a predefined set, tune its hyperparameters, and evaluate the model.

## Features

- Automatic data pre-processing using StandardScaler.
- Model selection from a predefined set (SVM, RandomForest, Logistic Regression).
- Automatic hyperparameter tuning using GridSearchCV.
- Model evaluation using accuracy score.

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/grausamkeit999/automl-system.git
    ```

2. Navigate into the cloned project directory:
    ```
    cd automl-system
    ```

3. Create a virtual environment:
    ```
    python -m venv env
    ```

4. Activate the virtual environment:
    - On Windows:
        ```
        env\Scripts\activate
        ```
    - On Unix or MacOS:
        ```
        source env/bin/activate
        ```

5. Install the required packages:
    ```
    pip install -r requirements.txt
    ```

## Usage

1. Activate the virtual environment:
    - On Windows:
        ```
        env\Scripts\activate
        ```
    - On Unix or MacOS:
        ```
        source env/bin/activate
        ```

2. Run the script:
    ```
    python automl.py
    ```

## Contributing

Please feel free to fork this repository, make some changes, and open a pull request. Issues are also welcome.

## License

This project is licensed under the terms of the MIT license.

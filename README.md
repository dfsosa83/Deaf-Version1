# Deaf-Version1
Predict forex market using advanced data analysis &amp; ensemble modeling. Features engineering, voting methods, hyperparameter tuning, and super learner model for accurate predictions.

## Description
This project focuses on predicting a target variable in the forex market using a comprehensive set of variables such as close-open-high. Leveraging a historical database captured in seconds, we conducted thorough descriptive analysis, followed by an augmented data strategy designed to provide additional information and enhance the model's understanding of extreme distributions.

To further optimize performance, extensive feature engineering was performed, including the creation of noise variables that enabled an effective voting process in subsequent stages. The voting method employed three distinct models: ridge regression, decision trees, and linear regression. Each model was subjected to rigorous hyperparameter tuning, accompanied by proper validation sets ensuring accurate outcome assessment against real-world data.

By selecting the most influential variables, we trained the three models individually, implementing advanced techniques to fine-tune their performance. Finally, a super learner model was constructed, leveraging the predictions from the aforementioned models to generate the ultimate outcome.

With a focus on data-driven decision making and ensemble modeling, this project offers insights into the predictive capabilities of the forex market, exhibiting a comprehensive approach to extract valuable information and build sophisticated models for accurate forecasting.

## Installation

1. Clone the repository:

git clone [https://github.com/your-username/project.git](https://github.com/dfsosa83/Deaf-Version1.git)


2. Navigate to the project directory:

cd project


3. Create and activate the conda environment:
- Using environment.yml:
  ```
  conda env create -f environment.yml
  conda activate myenv
  ```
- Using requirements.txt:
  ```
  conda create --name myenv --file requirements.txt
  conda activate myenv
  ```

4. Run the project:
python main.py


## Usage

Explain how to use your project in more detail. Include any necessary instructions or examples.

## Resources

Provide any additional resources or references related to your project.

- # ################################################ STEP FOR RUN #####################################

- # To install the environment using the .yml file:

Run the following command to create a new conda environment and install the required packages:

- conda env create -f environment.yml

- # To activate the environment:

Once the environment is successfully created, activate it using the following command:
On Windows: conda activate myenv
On macOS/Linux: source activate myenv
To install the environment using the requirements.txt file:

Run the following command to create a new conda environment and install the required packages:

- conda create --name myenv --file requirements.txt

- # o activate the environment:

Once the environment is successfully created, activate it using the following command:
- # On Windows: 
conda activate myenv

- # On macOS/Linux: 
source activate myenv

# FLASK
Inside conda environment:
flask --app C:/Users/david/OneDrive/Documents/deaf_reload/deaf_reload_flask run




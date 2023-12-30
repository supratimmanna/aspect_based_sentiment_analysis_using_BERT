# aspect_based_sentiment_analysisusing_BERT

This repo fine tunes a BERT model for topic prediction along with its sentiment.

> **Data**

There are different types of data available for fine tuning and the data location is: **_'data/ae/'_**.

For this fine tuning part we use the **_dataset/ae/laptop_** data. It contains several feedbacks for purchased laptops.

> **The model**

We use a pre-trained BERT model with an extra topic prediction layer on it. One can look into the **_topic_extraction.py_** file to get a detailed understanding of the model and its fine tuning process. 

> **How to fine tune the BERT model**

Run the **_run_topic_extraction.py_** file to fine tune the model.

Things to keep in mind while fine tuning:
```
* We need a pre-trained BERT model. 
* If one is having diffciulty to automatically download the model from huggingface while running the code then first download the model locally then give that downloaded path to the model.
```
Look into this Look into this [link]([https://www.anaconda.com/products/individual](https://huggingface.co/docs/transformers/training)) to find out more about how to fine tune a BERT model

> **Required variables to run the fune tuning**

All the required varibales can be found **_config/config.yaml_** file. Change the value accordingly.
go to [link](https://www.anaconda.com/products/individual)

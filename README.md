# Analyze your favourite Series with NLP
In this project, I analyzed a series using NLP and LLMs. I scraped my own dataset, used zero-shot classifiers, built my own LLM text classifier, used NER to build a character network, and created a character chatbot to chat with your favorite characters. In the end, I integrated everything into a web GUI using Gradio. This NLP project elevated my CV to another level, and I gained numerous NLP skills that are highly sought after in the market.

## Overview
In this project, I created 5 models, each containing the code for a different part of the project:
      
**crawler**: This folder contains the code for web scraping the internet to build a comprehensive dataset about the anime using Scrapy.

**character_network**: This folder contains the code for creating an intricate character network using Spacy's NER model, NetworkX, and PyViz.

**text_classifier**: This folder contains the code for training a text classifier that can classify text into multiple classes.

**theme_classifier**: This folder contains the code for extracting the main themes of the series using Zero-shot classifiers.

**charater_chat_bot**: This folder contains the code for building a charatcer chatbot with LLMs to chat with your favorite charaters from the series. 


## Requirements
Before running the code in this project, make sure you have installed all packages in the requirements.txt by running

```pip install -r requirements.txt```
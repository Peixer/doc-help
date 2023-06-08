
# LangChain Documentation Helper

A repository for learning LangChain by building a generative ai application.

This is a web application is using a Pinecone as a vectorsotre and answers questions about LangChain 
(sources from LangChain official documentation). 


![Logo](https://github.com/emarco177/documentation-helper/blob/main/static/banner.gif)
[![udemy](https://img.shields.io/badge/LangChain%20Udemy%20Course-%2412.99-green)](https://www.udemy.com/course/langchain/?couponCode=LANGCHAINCD8C0B4060)

## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`PINECONE_API_KEY`
`PINECONE_ENVIRONMENT_REGION`
`OPENAI_API_KEY`

## Run Locally

Clone the project

```bash
  git clone https://github.com/emarco177/documentation-helper.git
```

Go to the project directory

```bash
  cd documentation-helper
```

Download LangChain Documentation
```bash
  mkdir langchain-docs
  wget -r -A.html -P langchain-docs https://python.langchain.com/en/latest/index.html
```

Install dependencies

```bash
  pipenv install
```

Start the flask server

```bash
  streamlit run main.py
```


## Running Tests

To run tests, run the following command

```bash
  pipenv run pytest .
```


## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://www.udemy.com/course/langchain/?referralCode=D981B8213164A3EA91AC)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/eden-marco/)
[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://www.udemy.com/user/eden-marco/)



## Updated SetUp

- apt update
- git clone X
- apt install pipenv
- [INSTALL PYENV](https://brain2life.hashnode.dev/how-to-install-pyenv-python-version-manager-on-ubuntu-2004)
- pyenv install 3.11.0
    - sudo apt remove pipenv
    - pip3 install pipenv
    - python3 -m pipenv shell
    - pipenv install
- apt install nvidia-cudnn
- pipenv install torch

## Updated Run Locally
- pipenv run python ingestion.py (If needed)
- pipenv run streamlit run main.py 
- sudo ufw allow 8501
- sudo ufw allow 22
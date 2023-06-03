from typing import Any, Dict, List

from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY
from transformers import AutoTokenizer, pipeline
import torch


def load_model():
    """
    Select a model on huggingface.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.
    """
    model = "tiiuae/falcon-7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-xl", model_kwargs={"device": "cuda"}
    )

    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )

    llm = load_model()
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=db.as_retriever(), return_source_documents=True
    )
    return qa({"question": query, "chat_history": chat_history})

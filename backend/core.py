from typing import Any, Dict, List

from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chat_models import ChatOpenAI
from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
import torch


def load_model():
    """
    Select a model on huggingface.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.
    """
    # model = "tiiuae/falcon-7b-instruct"
    # tokenizer = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        'mosaicml/mpt-7b',
        trust_remote_code=True,
    )
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

def load_modelOpenAI():
    return ChatOpenAI(
        verbose=True,
        temperature=0,
    )

def load_model_vicuna():
    '''
    Select a model on huggingface. 
    If you are running this for the first time, it will download a model for you. 
    subsequent runs will use the model from the disk. 
    '''
    model_id = "TheBloke/vicuna-7B-1.1-HF"
    tokenizer = LlamaTokenizer.from_pretrained(model_id)

    model = LlamaForCausalLM.from_pretrained(model_id,
                                            #   load_in_8bit=True, # set these options if your GPU supports them!
                                            #   device_map=1#'auto',
                                            #   torch_dtype=torch.float16,
                                            #   low_cpu_mem_usage=True
                                              )

    pipe = pipeline(
        "text-generation",
        model=model, 
        tokenizer=tokenizer, 
        max_length=2048,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)

    return local_llm

def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-xl", model_kwargs={"device": "cpu"}
    )

    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )

    llm = load_model_vicuna()
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=db.as_retriever(), return_source_documents=True
    )
    return qa({"question": query, "chat_history": chat_history})

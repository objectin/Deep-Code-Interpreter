#%%
# %%
import os
import getpass
import gradio as gr

# %%
# !pip install --upgrade langchain "deeplake[enterprise]" openai tiktoken sentence-transformers huggingface_hub transformers

# %%
git_pj_name = "nautilus_trader"
subname = 'all-MiniLM-L12-v2_splitted'
root_dir = f"./{git_pj_name}"
username = "intuitionwith"  # replace with your username from app.activeloop.ai

with open('reader.key', 'r') as f:
    os.environ["OPENAI_API_KEY"] = f.readline().strip()
    activeloop_token = f.readline().strip()
    os.environ["ACTIVELOOP_TOKEN"] = activeloop_token

#%%
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import DeepLake

#embeddings = OpenAIEmbeddings(disallowed_special=())

embeddings_model_name = "sentence-transformers/all-MiniLM-L12-v2"
model_kwargs = {"device": "cuda"}

embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, model_kwargs=model_kwargs)

# %% [markdown]
# ### Using Vec DB

# %%
from langchain.vectorstores import DeepLake

# %%
db = DeepLake(
    dataset_path=f"hub://{username}/{git_pj_name+'_'+subname}",
    read_only=True,
    embedding=embeddings,
)

# %%
retriever = db.as_retriever()
# retriever.search_kwargs["distance_metric"] = "cos"
# retriever.search_kwargs["fetch_k"] = 100
# retriever.search_kwargs["maximal_marginal_relevance"] = True
# retriever.search_kwargs["k"] = 10

# %%
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

model = ChatOpenAI(model_name="gpt-4-32k", max_tokens=32000)  # switch to 'gpt-4'
qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)

# %%
chat_history = []

def ask_db(question):
    result = qa({"question": question, "chat_history": chat_history})
    # print(f"-> **Question**: {question} \n")
    # print(f"**Answer**: {result['answer']} \n")
    
    return (question, result["answer"])

#%%

def answer_from_db(text):
    global chat_history
    ret = ask_db(text)
    chat_history.append(ret)
    return ret[1]

with gr.Blocks() as demo:
    name = gr.Textbox(label="Code Questioner")
    output = gr.Textbox(label="Output Box")
    greet_btn = gr.Button("Answer")
    greet_btn.click(fn=answer_from_db, inputs=name, outputs=output, api_name="answer_from_db")


print(f"Testing hello world: {answer_from_db('What is nautilus trader?')}")

# %%
demo.queue()
demo.launch(share=True)
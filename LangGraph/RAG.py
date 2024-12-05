import warnings
from langchain_ollama import ChatOllama
import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import tools_condition, ToolNode
from PIL import Image as PILImage
from io import BytesIO
import json
from langgraph.graph import StateGraph, START, END , MessagesState
from langchain_core.messages import HumanMessage, SystemMessage



def display_graph(graph):
    image_data = graph.get_graph().draw_mermaid_png()
    image = PILImage.open(BytesIO(image_data))
    image.show()


# load a specific prompt from a given file
def load_prompt(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()


# Ignore all warnings
warnings.filterwarnings("ignore")

# Get the current directory dynamically
current_dir = os.getcwd()

# Construct file paths
input_file_path = os.path.join(current_dir, "Resources/TEST_PDF.pdf")
prompt_file_path = os.path.join(current_dir, "Prompts/new_info_extraction.txt")


#input_file_path = r"/home/eathanasakis/Thesis/RAG_Query/Resources/Thesis_Resources/PDFs/TEST_PDF.pdf"

embed_model = HuggingFaceEmbedding('paraphrase-multilingual-MiniLM-L12-v2')

splitter = SentenceSplitter(chunk_size=700)

llm = ChatOllama(model="llama3.1:70b", temperature = 0)

Settings.llm = llm
Settings.embed_model = embed_model


documents = SimpleDirectoryReader(
    input_files=[input_file_path]).load_data()

vector_store_index = VectorStoreIndex.from_documents(documents=documents,
                                                     splitter=splitter
                                                     )
                
query_engine = vector_store_index.as_query_engine()

# Tools
@tool
def query_tool(query: str):
    """
    Search for information given in the input arguments "query" 
    In the Vector Store.
    The vector store has been created from relevant documents.
    """
    result = query_engine.query(query)
    return result


@tool
def store_json(response):
    """
    Stores the json text into a file.
    """
    # dict_string = json.loads(str(response))
    # json_string = json.dumps(dict_string)

    # store it in a file
    output_json_file = os.path.join(current_dir, "outputs/json_output.txt")
    with open(output_json_file, 'w', encoding='utf-8') as file:
            return file.write(response)  

    #return json_string



tools = [query_tool, store_json]

llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(content="You are a helpful assistant tasked with responding to general questions \
                        and performing RAG using indexing in the a vector store index.")


# Node
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools=tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant", tools_condition
)
builder.add_edge("tools", "assistant")

react_graph = builder.compile()

#display_graph(react_graph)

prompt = load_prompt(prompt_file_path)

messages = [HumanMessage(content=prompt)]


messages = react_graph.invoke({'messages': messages})

for m in messages["messages"]:
    m.pretty_print()


import os
import qdrant_client
from llama_index.core import SimpleDirectoryReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex

data_path = "./data/"

image_metadata_dict = {}

# Iterate over the files in the directory specified by data_path
for file in os.listdir(data_path):
    if file.endswith(".txt"):  # Check if the file ends with .txt
        filename = file
        # Replace the .txt extension with .jpg to find the corresponding image
        img_path = data_path + file.replace(".txt", ".jpg")
        if os.path.exists(img_path):  # Check if the .jpg image exists
            image_metadata_dict[len(image_metadata_dict)] = {
                "filename": filename,
                "img_path": img_path
            }
        else:
            # Replace the .txt extension with .png to find the corresponding image
            img_path = data_path + file.replace(".txt", ".png")
            if os.path.exists(img_path):  # Check if the .png image exists
                image_metadata_dict[len(image_metadata_dict)] = {
                    "filename": filename,
                    "img_path": img_path
                }

# Print the dictionary containing metadata about images
print(image_metadata_dict)


# Create a local Qdrant vector store 网络不通畅。下载到本地
client = qdrant_client.QdrantClient(path="qdrant_d_0")

text_store = QdrantVectorStore(
    client=client, collection_name="text_collection_0"
)
image_store = QdrantVectorStore(
    client=client, collection_name="image_collection_0"
)
storage_context = StorageContext.from_defaults(
    vector_store=text_store, image_store=image_store
)

# nltk 是 Natural Language Toolkit 的缩写，是 NLP工具包之一，提供文本分词、词性标注、文本清洗、文本分析等工具。
# 下载的punkt包需要遵循这个格式 append_path+'/corpora/'+punkt,stopwards等包
import nltk
nltk.data.path.append('/xx/SimpleRAG/Multi-Modal-RAG-Pipeline-on-Images-and-Text-Locally/nltk_data')
# 图片embedding
import clip
model, preprocess = clip.load("./embedding-weight/ViT-B-32.pt", device="cuda", jit=False)

# 文本 embedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
local_embed_model = HuggingFaceEmbedding(model_name="./embedding-weight/bge-small-en-v1.5") 

# Create the MultiModal index
documents = SimpleDirectoryReader(data_path).load_data()
index = MultiModalVectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model= local_embed_model, 
)
# retriever 结果

test_query = "Where is space?"
# generate  retrieval results
retriever = index.as_retriever(similarity_top_k=1, image_similarity_top_k=1)
retrieval_results = retriever.retrieve(test_query)

retrieved_image = []
for res_node in retrieval_results:
    if isinstance(res_node.node, ImageNode):
        retrieved_image.append(res_node.node.metadata["file_path"])
    else:
        display_source_node(res_node, source_length=200)













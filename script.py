from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Indexing 
# Load the PDF document
loader = PyPDFLoader("ordinance01.pdf")

docs = loader.lazy_load()

# print(docs)
# print(len(docs))

# Split the document into smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# chunks = splitter.create_documents([docs[0].page_content])

chunks = splitter.split_documents(docs)

print(len(chunks))
print(chunks[0].page_content)

# Embeddings 
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# store
vector_store = FAISS.from_documents(chunks, embeddings)
# print(vector_store.index_to_docstore_id)
# print(vector_store.get_by_ids(["06d4e557-df48-4a0e-b804-02534d6be158"]))

# retrieval
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# print(retriever.invoke("What is the ordinance about?"))

# Augmentation
llm = ChatGroq(model="llama-3.3-70b-versatile")

prompt  = PromptTemplate(
    template="Your are a helpful assistant. Answer the question based on the following context only. If the context is insufficient , just say you don't know.:\n\nContext: {context}\n\nQuestion: {question}",\
    input_variables=["context", "question"]
)

question = "what is SUGC"
# retrieved_docs = retriever.invoke(question)

# context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

# temp = ""
# for doc in retrieved_docs:
#     temp += doc.page_content + "\n\n"
# print(temp)
# final_prompt = prompt.format(context=context_text, question=question)
# print(final_prompt)

# Generation 
# answer = llm.invoke(final_prompt)
# print(answer.content) 

# Building chains 


def format_docs(retrieved_docs):
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return context_text

parallel_chain = RunnableParallel({
    'context' : retriever | RunnableLambda(format_docs),
    'question' : RunnablePassthrough()
})
parallel_chain.invoke(question)

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

print(main_chain.invoke(question))
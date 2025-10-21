import PyPDF2
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import chainlit as cl
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv() 

groq_api_key = os.environ['GROQ_API_KEY']

llm_groq = ChatGroq(
            groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768",
                         temperature=0.2)


@cl.on_chat_start
async def on_chat_start():
    files = None 

    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a pdf file to begin!",
            accept=["application/pdf"],
            max_size_mb=100,
            timeout=180,
        ).send()

    file = files[0] 
    print(file) 
   
    elements = [
    cl.Image(name="image", display="inline", path="pic.jpg")
    ]
    
    msg = cl.Message(content=f"Processing `{file.name}`...",elements=elements)
    await msg.send()

   
    pdf = PyPDF2.PdfReader(file.path)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()
        

    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
    texts = text_splitter.split_text(pdf_text)

    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )
    
    message_history = ChatMessageHistory()
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_groq,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler()
    
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"] 

    text_elements = [] 
    
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]
        
        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"
    await cl.Message(content=answer, elements=text_elements).send()
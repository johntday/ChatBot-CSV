import streamlit as st
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain


class Chatbot:
    _template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
    Chat History:
    {chat_history}
    Follow-up entry: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    qa_template = """"Answer the question based on the context and chat history below. If the
    question cannot be answered using the information provided answer with "I don't know".

    Context: SAP Commerce Cloud is the world's leading B2C and B2B commerce solution. 
    SAP Commerce Cloud is a multi-tenant, cloud-based commerce platform that empowers brands to create intelligent, 
    unified buying experiences across all channels — mobile, social, web, and store.
    
    Chat History: {chat_history}

    Question: {question}
    """
    QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["question", "chat_history"])

    def __init__(self, model_name, temperature, vectors):
        self.model_name = model_name
        self.temperature = temperature
        self.vectors = vectors

    def conversational_chat(self, query):
        """
        Starts a conversational chat with a model via Langchain
        """
        chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name=self.model_name, temperature=self.temperature),
            # condense_question_prompt=self.CONDENSE_QUESTION_PROMPT,
            # qa_prompt=self.QA_PROMPT,
            retriever=self.vectors.as_retriever(),
            return_source_documents=True,
        )
        # my_chain = RetrievalQAWithSourcesChain.from_chain_type(
        #     llm=OpenAI(model_name=self.model_name, temperature=self.temperature),
        #     # condense_question_prompt=self.CONDENSE_QUESTION_PROMPT,
        #     # qa_prompt=self.QA_PROMPT,
        #     retriever=self.vectors.as_retriever(),
        # )
        # my_chain_result = my_chain({"question": query, "chat_history": st.session_state["history"]})
        # print(my_chain_result)

        result = chain({"question": query, "chat_history": st.session_state["history"]})
        # print("source_documents: " + result['source_documents'])
        print(*result['source_documents'], sep="\n\n")

        st.session_state["history"].append((query, result["answer"]))

        return result["answer"]

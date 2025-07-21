import streamlit as st
from src.utils import *
import os
from dotenv import load_dotenv

from langchain_openai import (
    AzureOpenAIEmbeddings,
    OpenAIEmbeddings,
    AzureChatOpenAI,
    ChatOpenAI
)
from langchain_core.messages import (
    HumanMessage, 
    AIMessage,
)
USER_NAME = "user"
ASSISTANT_NAME = "assistant"

def main():

    #　Title
    st.title('Azure Chat Demo')

    # emmbeddingsのモデルを取得
    embeddings = None
    if os.getenv('AZURE_OPENAI_API_KEY') != "":
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment="embedding",
            openai_api_version="2024-06-01"
        )
    elif os.getenv('OPENAI_API_KEY') != "":
        # OpenAIの場合
        embeddings = OpenAIEmbeddings()
    else:
        st.error("APIKeyの設定を確認してください")

    # chatのモデルを取得
    model = None
    # Azureの場合
    if os.getenv('AZURE_OPENAI_API_KEY') != "":
        model = AzureChatOpenAI(
            azure_deployment="chat",
            openai_api_version="2024-06-01",
        )
    elif os.getenv('OPENAI_API_KEY') != "":
        # OpenAIの場合
        model = ChatOpenAI(model="gpt-4")
    else:
        st.error("APIKeyの設定を確認してください")

    # Chain取得
    contextualize_chain = get_contextualize_prompt_chain(model)
    chain = get_chain(model)

    # FAISSからretrieverを取得
    retriever = pull_from_faiss(embeddings)

    # チャットログを保存したセッション情報を初期化
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []

    # ユーザーのメッセージ入力
    user_msg = st.chat_input("ここにメッセージを入力")

    if user_msg:
        
        # 以前のチャットログを表示
        for chat in st.session_state.chat_log:
            if isinstance(chat, AIMessage):
                with st.chat_message(ASSISTANT_NAME):
                    st.write(chat.content)
            else:
                with st.chat_message(USER_NAME):
                    st.write(chat.content)

        # ユーザーのメッセージを表示
        with st.chat_message(USER_NAME):
            st.write(user_msg)
            
        # 質問を修正する
        if st.session_state.chat_log:
            new_msg = contextualize_chain.invoke({"chat_history": st.session_state.chat_log, "input": user_msg})
        else:
            new_msg = user_msg
        print(user_msg, "=>", new_msg)

        # 類似ドキュメントを取得
        # ドキュメント数の調整はretriever取得時に設定
        relavant_docs = retriever.invoke(new_msg)

        # 質問の回答を表示 
        # response = chain.invoke({"chat_history": st.session_state.chat_log, "context": relavant_docs, "input": new_msg})
        # response = response.content
        # with st.chat_message(ASSISTANT_NAME):
        #     st.write(response)
        response = ""
        with st.chat_message(ASSISTANT_NAME):
            msg_placeholder = st.empty()
            
            for r in chain.stream({"chat_history": st.session_state.chat_log, "context": relavant_docs, "input": user_msg}):
                response += r.content
                msg_placeholder.markdown(response + "■")
            msg_placeholder.markdown(response)

        # セッションにチャットログを追加
        st.session_state.chat_log.extend([
            HumanMessage(content=user_msg),
            AIMessage(content=response)
        ])

if __name__ == '__main__':
    load_dotenv('./../.env')
    main()
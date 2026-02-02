import streamlit as st
from langchain_community.chat_models import ChatZhipuAI
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
import sys
sys.path.append("notebook/C3 搭建知识库") # 将父目录放入系统路径中
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain_chroma import Chroma

def get_retriever():
    # 定义 Embeddings
    embedding = ZhipuAIEmbeddings()
    # 向量数据库持久化路径
    persist_directory = 'data_base/vector_db/chroma2'
    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    return vectordb.as_retriever()

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs["context"])

def get_qa_history_chain():
    retriever = get_retriever()
    llm = ChatZhipuAI(model="glm-4", temperature=0.7, api_key=os.environ["ZHIPUAI_API_KEY"])
    condense_question_system_template = (
        "请根据聊天记录总结用户最近的问题，"
        "如果没有多余的聊天记录则返回用户的问题。"
    )
    condense_question_prompt = ChatPromptTemplate([
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])

    retrieve_docs = RunnableBranch(
        (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | retriever, ),
        condense_question_prompt | llm | StrOutputParser() | retriever,
    )

    system_prompt = (
        "你是一个问答任务的助手。 "
        "请使用检索到的上下文片段回答这个问题。 "
        "如果你不知道答案就说不知道。 "
        "请使用简洁的话语回答用户。"
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    qa_chain = (
        RunnablePassthrough().assign(context=combine_docs)
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    qa_history_chain = RunnablePassthrough().assign(
        context = retrieve_docs, 
        ).assign(answer=qa_chain)
    return qa_history_chain

def gen_response(chain, input, chat_history):
    response = chain.stream({
        "input": input,
        "chat_history": chat_history
    })
    for res in response:
        if "answer" in res.keys():
            yield res["answer"]

def main():
    # ============ 页面配置 ============
    st.set_page_config(
        page_title="琉璃海探索助手",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # ============ 自定义CSS样式 ============
    st.markdown("""
    <style>
    /* 主标题样式 */
    .main-title {
        background: linear-gradient(90deg, #0066CC 0%, #00CCFF 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(0, 102, 204, 0.3);
        text-align: center;
    }
    
    .main-title h1 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
    }
    
    .main-title p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 10px 0 0 0;
    }
    
    /* 欢迎卡片样式 */
    .welcome-card {
        background: linear-gradient(135deg, #E6F7FF 0%, #B3E0FF 100%);
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        border: 2px solid #66B3FF;
        box-shadow: 0 4px 8px rgba(102, 179, 255, 0.2);
    }
    
    .welcome-card h3 {
        color: #0066CC;
        margin-top: 0;
    }
    
    /* 侧边栏样式 */
    .sidebar-section {
        background: #F8FBFF;
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
        border: 1px solid #E6F2FF;
    }
    
    /* 聊天消息样式 */
    .stChatMessage {
        border-radius: 12px !important;
        margin: 10px 0;
    }
    
    /* 示例问题按钮 */
    .example-question {
        background: linear-gradient(135deg, #66B3FF 0%, #3399FF 100%);
        color: white;
        border: none;
        padding: 12px 20px;
        border-radius: 25px;
        margin: 8px 0;
        width: 100%;
        text-align: left;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    
    .example-question:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(51, 153, 255, 0.3);
        background: linear-gradient(135deg, #3399FF 0%, #0066CC 100%);
    }
    
    /* 输入框样式 */
    .stChatInput {
        border-radius: 25px !important;
        border: 2px solid #66B3FF !important;
    }
    
    /* 容器样式 */
    .chat-container {
        background: linear-gradient(180deg, #FFFFFF 0%, #F0F9FF 100%);
        border-radius: 15px;
        padding: 5px;
        border: 1px solid #E6F2FF;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # ============ 页面标题 ============
    st.markdown("""
    <div class="main-title">
        <h1>🌊 琉璃海探索助手</h1>
        <p>探索大洋深处的发光生命奇观 • 基于知识库的智能问答系统</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ============ 侧边栏 ============
    with st.sidebar:
        st.markdown("### 🔍 知识库信息")
        
        # 知识库统计
        with st.container():
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("**📚 文章内容**")
            st.markdown("""
            - 神秘的海洋荧光现象
            - 观测时机与全球热点
            - 文化意义与现代研究
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 对话统计
        if "messages" in st.session_state and st.session_state.messages:
            human_count = sum(1 for msg in st.session_state.messages if msg[0] == "human")
            ai_count = sum(1 for msg in st.session_state.messages if msg[0] == "ai")
            
            with st.container():
                st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
                st.markdown("**📊 对话统计**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("您的提问", human_count)
                with col2:
                    st.metric("AI回复", ai_count)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # 操作按钮
        with st.container():
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("**⚙️ 系统操作**")
            
            if st.button("🔄 清空对话历史", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
            
            if st.button("📥 导出对话记录", use_container_width=True):
                if st.session_state.messages:
                    from datetime import datetime
                    export_text = f"琉璃海探索助手对话记录\n生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    for role, content in st.session_state.messages:
                        export_text += f"{'您' if role == 'human' else 'AI助手'}: {content}\n\n"
                    st.download_button(
                        label="📥 下载对话记录",
                        data=export_text,
                        file_name=f"琉璃海对话_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 关于信息
        with st.container():
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("**ℹ️ 关于系统**")
            st.markdown("""
            - **模型**：ChatGLM-4
            - **知识库**：琉璃海专题文章
            - **版本**：v1.0.0
            """)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # ============ 主内容区 ============
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # ============ 初始化session state ============
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "qa_history_chain" not in st.session_state:
            with st.spinner("🔄 正在初始化智能问答系统..."):
                st.session_state.qa_history_chain = get_qa_history_chain()
        
        # ============ 聊天容器 ============
        with st.container():
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            messages = st.container(height=500)
            
            # 显示欢迎信息（仅首次）
            if len(st.session_state.messages) == 0:
                with messages:
                    st.markdown('<div class="welcome-card">', unsafe_allow_html=True)
                    st.markdown("### 🌟 欢迎探索琉璃海的奥秘")
                    st.markdown("""
                    **琉璃海**——大洋深处神秘的发光生命现象，是自然界最迷人的奇观之一。
                    
                    💡 **您可以向我询问：**
                    - 琉璃海的科学原理和形成机制
                    - 全球最佳观测地点和时间
                    - 相关的海洋生物和生态意义
                    - 文化传说和现代研究进展
                    - 环境保护现状和未来展望
                    
                    ⚡ **特色功能：**
                    - 基于知识库的精准回答
                    - 多轮对话理解上下文
                    - 流式输出实时响应
                    """)
                    
                    # 示例问题快速入口
                    st.markdown("### 🚀 快速开始")
                    example_questions = [
                        "琉璃海是什么？简要介绍一下",
                        "哪些地方可以看到琉璃海现象？",
                        "琉璃海是怎么形成的？科学原理是什么？",
                        "关于琉璃海有哪些有趣的传说故事？",
                        "现在的琉璃海面临什么环境威胁？"
                    ]
                    
                    for question in example_questions:
                        if st.button(
                            f"• {question}",
                            key=f"ex_{hash(question)}",
                            use_container_width=True
                        ):
                            if "auto_question" not in st.session_state:
                                st.session_state.auto_question = question
                            st.rerun()
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # ============ 显示对话历史 ============
            for message in st.session_state.messages:
                with messages.chat_message(message[0]):
                    # 为AI回复添加特殊样式
                    if message[0] == "ai":
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #F0F9FF 0%, #E6F7FF 100%);
                            padding: 15px;
                            border-radius: 10px;
                            border-left: 4px solid #0066CC;
                        ">
                            {message[1]}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.write(message[1])
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # ============ 右侧功能栏 ============
        st.markdown("### 📋 主题知识点")
        
        topic_points = {
            "🔬 科学探索": [
                "夜光藻的发光机制",
                "荧光素酶化学反应",
                "深海发光生物种类",
                "生态系统的意义"
            ],
            "📍 地理热点": [
                "波多黎各莫纳海峡",
                "马尔代夫星海沙滩",
                "日本富山湾",
                "澳大利亚吉普斯兰湖"
            ],
            "📜 文化传说": [
                "塔法伊的眼泪传说",
                "毛利人海洋精灵",
                "夏威夷祖先指引",
                "文学艺术中的身影"
            ],
            "🔍 现代研究": [
                "量子效率突破",
                "基因密码破译",
                "医疗应用前景",
                "环境保护计划"
            ]
        }
        
        for category, points in topic_points.items():
            with st.expander(category, expanded=True):
                for point in points:
                    if st.button(
                        f"• {point}",
                        key=f"topic_{hash(point)}",
                        use_container_width=True,
                        help=f"点击询问关于{point}的信息"
                    ):
                        if "auto_question" not in st.session_state:
                            st.session_state.auto_question = f"请介绍一下{point}"
                        st.rerun()
        
        # 快速操作
        st.markdown("### ⚡ 快速操作")
        if st.button("❓ 随机提问", use_container_width=True, help="随机生成一个关于琉璃海的问题"):
            import random
            random_questions = [
                "琉璃海对海洋生态系统有什么重要意义？",
                "观测琉璃海需要注意什么条件？",
                "琉璃海现象在古代有哪些文化记载？",
                "现代科学家如何研究琉璃海现象？",
                "琉璃海的发光颜色为什么大多是蓝绿色？"
            ]
            st.session_state.auto_question = random.choice(random_questions)
            st.rerun()
    
    # ============ 用户输入区域 ============
    st.markdown("---")
    
    # 检查是否有预设问题
    prompt = None
    if "auto_question" in st.session_state:
        prompt = st.session_state.pop("auto_question")
    
    # 用户输入
    user_input = st.chat_input(
        "💭 请输入您关于琉璃海的问题...",
        key="chat_input"
    )
    
    final_prompt = prompt or user_input
    
    if final_prompt:
        # 添加用户消息
        st.session_state.messages.append(("human", final_prompt))
        
        # 显示用户消息（立即显示）
        with messages.chat_message("human"):
            st.write(final_prompt)
        
        # 生成AI回复
        with st.spinner("🌊 AI正在查阅琉璃海知识库..."):
            try:
                answer_stream = gen_response(
                    chain=st.session_state.qa_history_chain,
                    input=final_prompt,
                    chat_history=st.session_state.messages
                )
                
                # 流式输出AI回复
                with messages.chat_message("ai"):
                    response_container = st.empty()
                    full_response = ""
                    
                    for chunk in answer_stream:
                        full_response += chunk
                        response_container.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #F0F9FF 0%, #E6F7FF 100%);
                            padding: 15px;
                            border-radius: 10px;
                            border-left: 4px solid #0066CC;
                        ">
                            {full_response}
                        </div>
                        """, unsafe_allow_html=True)
                
                # 保存完整回复到历史
                st.session_state.messages.append(("ai", full_response))
                
                # 自动滚动（通过rerun）
                st.rerun()
                
            except Exception as e:
                st.error(f"生成回复时出现错误: {str(e)}")
                error_msg = "抱歉，我在处理您的请求时遇到了问题。请稍后再试或尝试重新提问。"
                st.session_state.messages.append(("ai", error_msg))

if __name__ == "__main__":
    main()

# def main():
#     # 添加CSS样式
#     st.markdown("""
#     <style>
#     .chat-container {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         border-radius: 10px;
#         padding: 20px;
#         margin-bottom: 20px;
#     }
#     .welcome-message {
#         background-color: #f0f2f6;
#         padding: 15px;
#         border-radius: 10px;
#         margin: 10px 0;
#     }
#     </style>
#     """, unsafe_allow_html=True)
    
#     st.markdown('### 🦜🔗 动手学大模型应用开发')
#     # st.session_state可以存储用户与应用交互期间的状态与数据
    
#     # 初始化
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
    
#     # 存储检索问答链
#     if "qa_history_chain" not in st.session_state:
#         st.session_state.qa_history_chain = get_qa_history_chain()
        
#     # 建立容器 高度为500 px
#     messages = st.container(height=550)

#     # 如果对话为空，显示欢迎信息
#     if len(st.session_state.messages) == 0:
#         with messages:
#             st.markdown('<div class="chat-container">', unsafe_allow_html=True)
#             st.markdown('<div class="welcome-message">', unsafe_allow_html=True)
#             st.markdown("### 🤖 欢迎使用智能问答系统")
#             st.markdown("""
#             - 💬 您可以在这里询问关于“琉璃海”的问题
#             - 🔍 系统会自动检索相关知识
#             - ⚡ 支持多轮对话
#             - 📚 基于最新的大语言模型技术
#             """)
#             st.markdown('</div>', unsafe_allow_html=True)
#             st.markdown('</div>', unsafe_allow_html=True)
    
#     # 显示整个对话历史
#     for message in st.session_state.messages: # 遍历对话历史
#             with messages.chat_message(message[0]): # messages指在容器下显示，chat_message显示用户及ai头像
#                 st.write(message[1]) # 打印内容

#     # 用户输入
#     if prompt := st.chat_input("Say something"):
#         # 将用户输入添加到对话历史中
#         st.session_state.messages.append(("human", prompt))
#         # 显示当前用户输入
#         with messages.chat_message("human"):
#             st.write(prompt)
#         # 生成回复
#         answer = gen_response(
#             chain=st.session_state.qa_history_chain,
#             input=prompt,
#             chat_history=st.session_state.messages
#         )
#         # 流式输出
#         with messages.chat_message("ai"):
#             output = st.write_stream(answer)
#         # 将输出存入st.session_state.messages
#         st.session_state.messages.append(("ai", output))

# if __name__ == "__main__":
#     main()





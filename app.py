import streamlit as st
import os
from rag_chain import EoraRAGChain


@st.cache_resource
def load_rag_chain():
    """Инициализация и кэширование RAG chain"""
    chain = EoraRAGChain()
    data_path = os.getenv("DATA_PATH", "./data")

    with st.spinner("Загрузка документов..."):
        doc_count = chain.load_documents(data_path, include_web=True)
        if doc_count > 0:
            st.success(f"Загружено {doc_count} документов")
        else:
            st.warning("Документы не найдены, работаем только с веб-данными")

    return chain


def format_sources(sources, complexity_level):
    """Форматирование источников в зависимости от уровня сложности"""
    if complexity_level == "easy":
        return ""

    source_list = []
    for i, source in enumerate(sources, 1):
        source_name = source.get("source_file", source.get("url", f"Источник {i}"))
        source_list.append(f"[{i}] {source_name}")

    if complexity_level == "medium":
        return f"\n\nИсточники: {', '.join([f'[{i}]' for i in range(1, len(sources) + 1)])}"

    return ""


def main():
    st.set_page_config(page_title="EORA AI Assistant", layout="wide")

    st.title("🤖 EORA AI Assistant")
    st.markdown("Задайте вопрос о наших проектах и услугах")

    if not os.getenv("OPENAI_API_KEY"):
        st.error("Необходимо установить OPENAI_API_KEY в переменных окружения")
        st.stop()

    try:
        rag_chain = load_rag_chain()
    except Exception as e:
        st.error(f"Ошибка при инициализации: {e}")
        st.stop()

    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("Настройки")
        complexity_level = st.selectbox(
            "Уровень сложности ответа:",
            options=["easy", "medium", "hard"],
            format_func=lambda x: {
                "easy": "Простой",
                "medium": "Со списком источников",
                "hard": "С inline ссылками",
            }[x],
        )

        st.markdown("---")
        st.markdown("**Примеры вопросов:**")
        st.markdown("- Что вы можете сделать для ритейлеров?")
        st.markdown("- Какие проекты вы делали для Магнита?")
        st.markdown("- Расскажите о ваших AI решениях")

    with col1:
        st.subheader("Чат")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and message.get("sources"):
                    with st.expander("Источники", expanded=False):
                        for i, source in enumerate(message["sources"], 1):
                            source_name = source.get(
                                "source_file", source.get("url", "Неизвестный источник")
                            )
                            st.write(f"**[{i}]** {source_name}")

        if prompt := st.chat_input("Ваш вопрос:"):
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Генерирую ответ..."):
                    try:
                        result = rag_chain.generate_answer(prompt, complexity_level)

                        formatted_answer = result["answer"]
                        if complexity_level == "medium" and result["sources"]:
                            formatted_answer += format_sources(
                                result["sources"], complexity_level
                            )

                        st.markdown(formatted_answer)

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": formatted_answer,
                                "sources": result["sources"],
                            }
                        )

                        if result["sources"] and complexity_level != "easy":
                            with st.expander("Источники", expanded=False):
                                for i, source in enumerate(result["sources"], 1):
                                    source_name = source.get(
                                        "source_file",
                                        source.get("url", "Неизвестный источник"),
                                    )
                                    st.write(f"**[{i}]** {source_name}")

                    except Exception as e:
                        error_msg = f"Ошибка при генерации ответа: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_msg, "sources": []}
                        )


if __name__ == "__main__":
    main()

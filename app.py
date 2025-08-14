import streamlit as st
import os
from core.rag_chain import EoraRAGChain
from core.config import Config
from core.exceptions import ConfigurationError, DocumentLoadError, LLMError
from utils.error_handler import ErrorHandler

Config.setup_logging()


@st.cache_resource
def load_rag_chain():
    """Инициализация и кэширование RAG chain"""
    Config.validate()
    chain = EoraRAGChain()
    data_path = Config.DATA_PATH

    with st.spinner("Загрузка документов..."):
        doc_count = chain.load_documents(data_path, include_web=True)
        if doc_count > 0:
            st.success(f"Загружено {doc_count} документов")
        else:
            st.warning("Документы не найдены, работаем только с веб-данными")

    return chain


@st.cache_data(ttl=3600)
def get_example_questions():
    """Кэшированные примеры вопросов"""
    return [
        "Что вы можете сделать для ритейлеров?",
        "Расскажите про HR-бота для Магнита",
        "Какие проекты вы делали для KazanExpress?",
        "Что такое поиск по картинкам для одежды?",
        "Какие чат-боты вы разрабатывали?",
        "Расскажите про проекты с компьютерным зрением",
        "Какие решения для промышленности вы создавали?",
        "Что вы делали для Dodo Pizza?",
    ]


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
    except ConfigurationError as e:
        st.error(f"Ошибка конфигурации: {e}")
        ErrorHandler.log_and_raise(e, "Configuration error")
        st.stop()
    except DocumentLoadError as e:
        st.error(f"Ошибка загрузки документов: {e}")
        ErrorHandler.log_and_raise(e, "Document load error")
        st.stop()
    except Exception as e:
        st.error(f"Неожиданная ошибка при инициализации: {e}")
        ErrorHandler.log_and_raise(e, "Unexpected initialization error")
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
        example_questions = get_example_questions()
        for i, question in enumerate(example_questions):
            if st.button(question, key=f"example_{i}", use_container_width=True):
                st.session_state.example_question = question
                st.rerun()

        st.markdown("---")
        st.markdown("**Статистика:**")
        if "messages" in st.session_state:
            st.metric("Сообщений в чате", len(st.session_state.messages))

        # Мониторинг производительности
        try:
            from utils.performance import PerformanceMonitor

            memory_usage = PerformanceMonitor.track_memory_usage()
            if memory_usage:
                st.metric("Память (MB)", f"{memory_usage:.1f}")
        except Exception:
            pass

        # Кнопка очистки чата
        if st.button("🗑️ Очистить чат", use_container_width=True):
            if "messages" in st.session_state:
                st.session_state.messages = []
            st.rerun()

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

        # Обработка примера вопроса
        if "example_question" in st.session_state:
            prompt = st.session_state.example_question
            del st.session_state.example_question
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()

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

                    except LLMError as e:
                        error_msg = f"Ошибка LLM: {str(e)}"
                        st.error(error_msg)
                        ErrorHandler.log_and_raise(e, "LLM error")
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_msg, "sources": []}
                        )
                    except Exception as e:
                        error_msg = f"Неожиданная ошибка при генерации ответа: {str(e)}"
                        st.error(error_msg)
                        ErrorHandler.log_and_raise(
                            e, "Unexpected answer generation error"
                        )
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_msg, "sources": []}
                        )


if __name__ == "__main__":
    main()

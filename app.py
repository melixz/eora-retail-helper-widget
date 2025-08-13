import streamlit as st


def main():
    st.title("EORA AI Assistant")
    st.write("Задайте вопрос о наших проектах и услугах")

    user_question = st.text_input("Ваш вопрос:")

    if user_question:
        st.write()


if __name__ == "__main__":
    main()

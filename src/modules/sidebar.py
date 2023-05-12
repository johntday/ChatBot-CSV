import json

import streamlit as st


class Sidebar:
    MODEL_OPTIONS = ["gpt-3.5-turbo", "gpt-4"]
    TEMPERATURE_MIN_VALUE = 0.0
    TEMPERATURE_MAX_VALUE = 1.0
    TEMPERATURE_DEFAULT_VALUE = 0.0
    TEMPERATURE_STEP = 0.01

    @staticmethod
    def about(app_name: str):
        about = st.sidebar.expander("🤖 About")
        sections = [
            f"- {app_name} is an AI chatbot with Hybris knowledge.  It has been trained on the following sources of Hybris documentation: "
            "[SAP Hybris Help v2211](https://help.sap.com/docs/SAP_COMMERCE_CLOUD_PUBLIC_CLOUD?version=v2211), "
            "[Hybrismart Articles](https://hybrismart.com), "
            "[Coveo](https://docs.coveo.com/en/), "
            "[CX Works](https://www.sap.com/cxworks/expert-recommendations/articles/commerce-cloud.html), "
            "[Worldpay SAP Hybris Addon](https://github.com/Worldpay/hybris)",
            "- The app is created with the following Python packages: "
            "[Langchain](https://github.com/hwchase17/langchain), "
            "[OpenAI](https://platform.openai.com/docs/models/gpt-3-5) and "
            "[Streamlit](https://github.com/streamlit/streamlit)",
            "- The app is inspired by [yvann-hub/ChatBot-CSV](https://github.com/yvann-hub/ChatBot-CSV)",
        ]
        for section in sections:
            about.write(section)

    # fixme: create usage

    @staticmethod
    def reset_chat_button():
        if st.button("Reset chat"):
            st.session_state["reset_chat"] = True
        st.session_state.setdefault("reset_chat", False)

    def model_selector(self):
        model = st.selectbox(label="Model", options=self.MODEL_OPTIONS)
        st.session_state["model"] = model

    def temperature_slider(self):
        temperature = st.slider(
            label="Temperature",
            min_value=self.TEMPERATURE_MIN_VALUE,
            max_value=self.TEMPERATURE_MAX_VALUE,
            value=self.TEMPERATURE_DEFAULT_VALUE,
            step=self.TEMPERATURE_STEP,
        )
        st.session_state["temperature"] = temperature

    def csv_agent_button(self):
        st.session_state.setdefault("show_csv_agent", False)
        if st.sidebar.button("CSV Agent"):
            st.session_state["show_csv_agent"] = not st.session_state["show_csv_agent"]

    def show_options(self):
        with st.sidebar.expander("🛠️ Settings", expanded=False):
            self.reset_chat_button()
            # self.csv_agent_button()
            self.model_selector()
            self.temperature_slider()
            st.session_state.setdefault("model", self.MODEL_OPTIONS[0])
            st.session_state.setdefault("temperature", self.TEMPERATURE_DEFAULT_VALUE)

    def show_sources(self, chat_sources):
        sources = st.sidebar.expander("🛠️ Top Sources for Answer", expanded=True)
        sources.empty()
        i = 0
        for chat_source in chat_sources:
            i += 1
            page_content = chat_source.page_content
            metadata = chat_source.metadata
            # text = page_content.replace("\n", " ").replace("\ue05c", " ").replace("\x00", "").replace("\'", "'")
            text = page_content

            sources.write(f"### Source Fragment {i}")
            sources.write(f"[{metadata['title']}]({metadata['source']}) \
                was published {metadata['published']} \
                by {metadata['source id']}")
            # if metadata['version'] != "":
            #     sources.write(f"Version: {metadata['version']}")
            sources.write(text)
            sources.write()



"""    def csv_agent(self, ):
            # Ajout du bouton pour naviguer vers la page du chatbot supplémentaire
        if csv_agent_button = st.sidebar.button("CSV Agent"):
            query = st.text_input(label="Use CSV agent for precise informations about the csv file itself")

            if query != "" :
                agent = create_csv_agent(ChatOpenAI(temperature=0), 'poto-associations-sample.csv', verbose=True)
                st.write(agent.run(query))"""

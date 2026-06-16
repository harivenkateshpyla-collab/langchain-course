from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

import os

load_dotenv()

def main():
    print("Hello from langchain-course!")
    information = """ Narendra Damodardas Modi[a] (born 17 September 1950) is an Indian politician who has served as the prime minister of India since 26 May 2014. Modi was the chief minister of Gujarat from 2001 to 2014 and is the Member of Parliament (MP) for Varanasi. He is a member of the Bharatiya Janata Party (BJP) and of the Rashtriya Swayamsevak Sangh (RSS), a right-wing Hindutva paramilitary volunteer organisation. He is India's third-longest-serving prime minister, and the longest-serving prime minister outside the Indian National Congress.[b]

Modi was born and raised in Vadnagar, where he completed his secondary education. He was introduced to the RSS at the age of eight, becoming a full-time worker for the organisation in Gujarat in 1971. The RSS assigned him to the BJP in 1985, and he rose through the party hierarchy, becoming general secretary in 1998.[c] In 2001, Modi was appointed chief minister of Gujarat and elected to the legislative assembly soon after. His administration is considered complicit in the 2002 Gujarat violence[d] in which over 1,000 people, mostly Muslim, were killed, with many others raped or mutilated.[e] An investigation authorised by the Supreme Court found no evidence to prosecute Modi.[f] While his policies as chief minister were credited for encouraging economic growth, his administration was criticised for failing to significantly improve health, poverty and education indices in the state.[g]
"""
    summary_template = """
    give a summary of the information: {information} about a person I want you to create:
    1. A short summary of the person in 2-3 sentences.
    2. Two interesting facts about the person.
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    # llm = ChatOpenAI(
    #     model_name="gpt-3.5-turbo",
    #     temperature=0,
    #     openai_api_key=os.environ.get("OLLAMA_API_KEY"),
    # )

    #llm = ChatOpenAI(temperature=0, model = "gpt-5", openai_api_key=os.environ.get("OLLAMA_API_KEY"))
    llm = ChatOllama(
        temperature=0,
        model="gemma3:1b ",    
    )
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)
if __name__ == "__main__":
    main()



from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel ,RunnablePassthrough

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature=0.8
)

prompt1 = PromptTemplate(
    template = "generate a joke about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = "explain the joke {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()


joke_gen_ai = RunnableSequence(prompt1, llm, parser)
parallel_chain = RunnableParallel({
    'joke' :RunnablePassthrough(),
    'explaination' : RunnableSequence(prompt2, llm, parser )
})


final_chain = RunnableSequence(joke_gen_ai, parallel_chain)
print(final_chain.invoke({'topic' : "ai"}))






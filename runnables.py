from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature=0.8
)

prompt1 = PromptTemplate(
    template = "generate a tweet about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = "generate a linkedin post about {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()


parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, llm, parser),
    'linkedin': RunnableSequence(prompt2, llm, parser)
})

print(parallel_chain.invoke({'topic' : 'ai'}))




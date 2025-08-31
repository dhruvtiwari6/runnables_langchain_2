from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnableLambda, RunnablePassthrough, RunnableBranch

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature=0.8
)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "wrtie a detailed report about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = "summarize the following text : \n {topic}",
    input_variables=['topic']
)

report_generation_chain = RunnableSequence(prompt1, llm , parser)
branch_chain = RunnableBranch(
    (lambda x : len(x.split())>500, RunnableSequence(prompt2, llm, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_generation_chain, branch_chain)
print(final_chain.invoke({'topic' : 'Russia vs ukraine war'}))




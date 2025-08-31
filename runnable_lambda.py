from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnableLambda, RunnablePassthrough

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature=0.8
)

def word_counter(text):
    return len(text.split())

runnable_word_counter = RunnableLambda(word_counter)
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "generate a joke about {topic}",
    input_variables=['topic']
)

joke_gen_ai = RunnableSequence(prompt1, llm, parser)
parallel_chain = RunnableParallel({
    'joke' :RunnablePassthrough(),
    'word_count' : runnable_word_counter
})

final_chain = RunnableSequence(joke_gen_ai, parallel_chain)


print(final_chain.invoke({'topic' : 'ai'}))




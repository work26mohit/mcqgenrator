import os
import certifi
from dotenv import load_dotenv
os.environ["SSL_CERT_FILE"] = certifi.where()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

primary_llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=api_key,
    temperature=0.7,
    timeout=20,
    max_retries=1,
)
fallback_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=api_key,
    temperature=0.7,
    timeout=20,
    max_retries=1,
)

llm = primary_llm.with_fallbacks([fallback_llm])

# 2. JSON-enforced LLMs with 20s timeout
primary_json_llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=api_key,
    temperature=0.7,
    timeout=20,
    max_retries=1,
    model_kwargs={"response_mime_type": "application/json"},
)
fallback_json_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=api_key,
    temperature=0.7,
    timeout=20,
    max_retries=1,
    model_kwargs={"response_mime_type": "application/json"},
)

json_llm = primary_json_llm.with_fallbacks([fallback_json_llm])

# 3. Prompt Templates
TEMPLATE = """
Text: {text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz of {number} multiple choice questions for {subject} students in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like RESPONSE_JSON below and use it as a guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}
"""

quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=TEMPLATE
)

TEMPLATE2 = """
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students. \
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
if the quiz is not at per with the cognitive and analytical abilities of the students, \
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

quiz_evaluation_prompt = PromptTemplate(
    input_variables=["subject", "quiz"],
    template=TEMPLATE2
)

# 4. Modern LCEL Chains (Replaces LLMChain & SequentialChain)
quiz_chain = quiz_generation_prompt | json_llm | StrOutputParser()
review_chain = quiz_evaluation_prompt | llm | StrOutputParser()

generate_evaluate_chain = (
    RunnablePassthrough.assign(quiz=quiz_chain)
    | RunnablePassthrough.assign(review=review_chain)
)
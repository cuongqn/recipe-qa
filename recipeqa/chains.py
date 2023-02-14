import langchain.chains as chains
import langchain.llms as llms
import langchain.prompts as prompts


llm = llms.OpenAI(
    temperature=0.1,
)

_distill_prompt = prompts.PromptTemplate(
    input_variables=["question", "context_str"],
    template="Provide relevant information that will help answer a question from the context below. "
    "Summarize the information in an unbiased tone. Use direct quotes "
    "where possible. Do not directly answer the question. "
    'Reply with "Not applicable" if the context is irrelevant to the question. '
    "Use 35 or less words."
    "\n\n"
    "{context_str}\n"
    "\n"
    "Question: {question}\n"
    "Relevant Information Summary: ",
)
distill_chain = chains.LLMChain(llm=llm, prompt=_distill_prompt)

_qa_prompt_str_chain = (
    "Write a comprehensive answer ({length}) "
    "for the question below solely based on the provided context. "
    "If the context is insufficient "
    'answer, reply "I cannot answer". '
    "For each sentence in your answer, indicate which sources most support it "
    "via valid citation markers at the end of sentences, like (Foo2012). "
    "Answer in an unbiased and balanced. Don't use prior knowledge."
    "Try to use the direct quotes, if present, from the context. "
    # "write a complete unbiased answer prefixed by \"Answer:\""
    "\n--------------------\n"
    "{context_str}\n"
    "----------------------\n"
    "Question: {question}\n"
    "Answer: "
)
_qa_prompt = prompts.PromptTemplate(
    input_variables=["question", "context_str", "length"],
    template=_qa_prompt_str_chain,
)
qa_chain = chains.LLMChain(llm=llm, prompt=_qa_prompt)


_query_prompt = prompts.PromptTemplate(
    input_variables=["question"],
    template="I would like to find recipes to answer this question: {question}. "
    'A search query that would bring up recipes related to this answer would be: "',
)
query_chain = chains.LLMChain(
    llm=llms.OpenAI(
        temperature=0.1,
        model_kwargs={"stop": ['"']},
    ),
    prompt=_query_prompt,
)


_edit_prompt = prompts.PromptTemplate(
    input_variables=["question", "answer"],
    template="The original question is: {question} "
    "We have been provided the following answer: {answer} "
    "Part of it may be truncated, please edit the answer to make it complete. "
    "If it appears to be complete, repeat it unchanged.\n\n",
)
edit_chain = chains.LLMChain(llm=llm, prompt=_edit_prompt)


_refine_prompt = prompts.PromptTemplate(
    input_variables=["question", "existing_answer", "context_str"],
    template="The original question is as follows: {question}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_str}\n"
    "------------\n"
    "Given the new context, refine the original answer to better answer the question. "
    "For each sentence in your refined answer, indicate which sources most support it "
    "via valid citation markers at the end of sentences, like (Foo2012). "
    "If the context isn't useful, return the original answer.",
)
refine_chain = chains.LLMChain(llm=llm, prompt=_refine_prompt)

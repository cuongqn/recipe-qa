# recipe-qa
LLM-powered Question-Answering for food and recipes based on [paper-qa](https://github.com/whitead/paper-qa)
## Installation
Clone and install locally
```bash
git clone https://github.com/cuongqn/recipe-qa.git
cd recipe-qa
pip install .
```

## Running
Set your OPENAI_API_KEY environment variable
```bash
export OPENAI_API_KEY=<key>
```
Use main as the entry point
```bash
python main.py -q "Can I combine miso paste with brussel sprouts?"
```
## Example
**Question**: Can I combine miso paste with brussel sprouts?

**Answer**: Yes, miso paste can be combined with brussel sprouts to make a delicious side dish. The Foodie Physician's recipe for Miso-Roasted-Brussels-Sprouts-2266030 provides a great example of how to do this. The recipe calls for roasting the sprouts with oil, salt, and pepper first to get them crispy. Then, the sprouts are tossed with the miso glaze and returned to the oven for a few minutes to caramelize. It is important to keep an eye on them as the glaze can burn quickly (The Foodie Physician-Miso-Roasted-Brussels-Sprouts-2266030).

Miso paste can also be used in other recipes that include brussel sprouts. For example, Little Spice Jar's Kale-Shaved-Brussel-Sprout-Salad-With-Ginger-Miso-Dressing-9437915 combines chopped kale and shaved Brussels sprouts with roasted peanuts, shredded cabbage, and leftover chicken or turkey, topped with a ginger miso dressing. The dressing is made with miso paste, ginger paste, lime juice, sesame oil, rice vinegar, oil, sugar,


**References**:

1. The Foodie Physician-Miso-Roasted-Brussels-Sprouts-2266030: https://thefoodiephysician.com/dining-with-doc-miso-roasted-brussels/

2. Little Spice Jar-Kale-Shaved-Brussel-Sprout-Salad-With-Ginger-Miso-Dressing-9437915: http://littlespicejar.com/asian-kale-shaved-brussel-sprout-salad-ginger-miso-dressing/

## Usage
#### Using `Agent` class directly
You can use the `Agent` class to directly query questions
```python
from recipeqa import recipe

agent = recipe.Agent()
answer = agent.query("Can I combine miso paste with brussel sprouts?")
```
#### Extending `Agent` class
Alternatively, the `Agent` class can be extended by using custom `Fetcher`, `Ranker`, `Summarizer`, and `qa_chain`. For example, to use a custom `Summarizer` and `qa_chain`, we need to do following
```python
import langchain.prompts as prompts
import langchain.chains as chains
import langchain.llms as llms

from recipeqa import summarizer
from recipeqa import recipe

# Create custom Summarizer class
class CustomSummarizer(summarizer.Summarizer):
    def __call__(self, docs: List[Document], **kwargs) -> str:
        # New method for summarizing documents
        return summarized_docs

# Create custom QA chain
custom_qa_prompt = prompts.PromptTemplate(
    input_variables=["question", "context_str", "length"],
    template="Write a comprehensive answer ({length}) "
    "for the question below solely based on the provided context. "
    "\n--------------------\n"
    "{context_str}\n"
    "----------------------\n"
    "Question: {question}\n"
    "Answer in italian: "
)
custom_qa_chain = chains.LLMChain(
    llm=llms.OpenAI(temperature=0.1),
    prompt=custom_qa_prompt,
)

# Instantiate Agent
agent = recipe.Agent(
    doc_summarizer=CustomSummarizer(),
    qa_chain=custom_qa_chain
)
answer = agent.query("Can I combine miso paste with brussel sprouts?")
```
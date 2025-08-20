from langchain_community.llms import OpenAI # type: ignore
from langchain.chains import LLMChain # type: ignore
from langchain.prompts import PromptTemplate # type: ignore
from dotenv import load_dotenv # type: ignore

load_dotenv()

def generate_pet_name(animal_type, pet_color):
    llm=OpenAI(temperature=0.7)

    # Prompt templates
    prompt_template_name = PromptTemplate(
        input_variables=["animal_type","pet_color"],
        template="I have a {animal_type} name and I want a cool name for it it is {pet_color} in color." \
        "Suggest me 10 cool names."
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name)
    response = name_chain.run({'animal_type': animal_type, 'pet_color': pet_color})
    return response

if __name__ == "__main__":
    print(generate_pet_name('cow', 'black'))

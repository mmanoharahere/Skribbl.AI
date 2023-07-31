from typing import Optional
from langchain import OpenAI
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools import BaseTool
from urllib import request
from PIL import Image
import re
import numpy as np

import requests
import openai
import os
from apikey import apikey
from prompt_gen import generate_prompt

os.environ['OPENAI_API_KEY'] = apikey
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
llm = OpenAI(temperature=0,openai_api_key=OPENAI_API_KEY,max_tokens=1500)

class MyImageGenTool(BaseTool):
    name = "GenerateImage"
    description = "Useful for when there is a need to generate an image." \
                  "Input: A prompt describing the image " \
                  "Output: Only the url of the generated image"

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.Image.create(
            prompt=query,
            n=1,
            size="512x512",
        )

        return response["data"][0]["url"]

    async def _arun(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        raise NotImplementedError("Does not support async")


def url_find(string):
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex, string)
    return [x[0] for x in url]


class ImgGen:
    prompt = None
    def __init__(self):
        pass
    
    @classmethod
    def generate_input(cls):

        tools = [MyImageGenTool()]

        mrkl = initialize_agent(
            tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False
        )
        ImgGen.prompt = generate_prompt()

        #print("---------- PROMPT --------: ", ImgGen.prompt)

        output = mrkl.run(ImgGen.prompt)

        #print(output)

        image_url = url_find(output)
        #print(image_url)
        response = requests.get(image_url[0], stream=True)
        img = Image.open(response.raw)
        return img
    
    #@staticmethod
    #def get_prompt():
    #    return ImgGen.prompt

class TextSimilarityTool(BaseTool):
    name="TextSimilarity"
    description="Check the correctness of the prompt."   

    def _run(self, prompt: str, guess: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        resp = openai.Embedding.create(
            input=[prompt, guess],
            engine="text-similarity-davinci-001")
        return resp['data']
    
    async def _arun(self, prompt: str, guess: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        raise NotImplementedError("Does not support async")
    
def handle_guess(guess: str)->Image:
    #print('------ GUESS -------:', guess)

    prompt = ImgGen.prompt
    #print('------- PROMPT -------:', prompt)
    resp = openai.Embedding.create(
            input=[prompt, guess],
            engine="text-similarity-davinci-001")
    
    embedding_a = resp['data'][0]['embedding']
    embedding_b = resp['data'][1]['embedding']
    similarity  = np.dot(embedding_a, embedding_b)
    
    #print(similarity)
    return similarity * 100

                                    
import gradio as gr
from utils import handle_input


with gr.Blocks(css=".gradio-container {background-color:HoneyDew }  {text-align: center; border: 3px solid green}") as demo:
#with gr.Blocks(theme=gr.themes.Glass()) as demo:
    with gr.Row():
        with gr.Column():
            #img_gen = ImgGen()
            output_block = gr.components.Image(label='Guess', type='pil').style(height=400, width=400)
            play = gr.Button(value="Play!")
            play.click(ImgGen.generate_input, inputs=None, outputs=output_block, api_name="SkribblAI")#,_js="window.location.reload()"
            
            guess=gr.components.Textbox(label= "Guess your prompt here")
            guess_btn = gr.Button(value="Check!")
            result = gr.components.Textbox(label= "Your prompt matches the AI prompt :")
            #result = gr.outputs.Number()
            guess_btn.click(handle_guess, inputs=guess, outputs=result,  api_name="SkribblAIGuess")
            try_agn = gr.Button(value="Try Again!")
            try_agn.click(None, _js="window.location.reload()")

            



if __name__ == "__main__":
    demo.launch()
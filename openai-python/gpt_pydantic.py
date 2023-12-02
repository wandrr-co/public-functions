import json
import os
from datetime import datetime

import openai
from pydantic import BaseModel
from typing import Any
from typing import Dict


# Define your Pydantic response model
## this tells the API HOW to respond to your prompt
## making it easier to anticipate, and code for, the data
## you''ll receive from OpenAI's API
## The expected response structure is derived from this dataset:
## https://www.kaggle.com/datasets/rounakbanik/pokemon
class PokemonOpenAIContract(BaseModel):
    name: str
    japanese_name: str
    pokedex_number: str
    percentage_male: str
    type1: str
    type2: str
    classification: str
    height_m: str
    weight_kg: str
    capture_rate: str
    base_egg_steps: str
    abilities: str
    experience_growth: str
    base_happiness: str
    against_: str
    hp: str
    attack: str
    defense: str
    sp_attack: str
    sp_defense: str
    speed: str
    generation: str
    is_legendary: str


# Define the OpenAI API client class
## This class contains the necessary methods to 
## interact with the OpenAI API and defines
## the method used to retrieve each Pokemon's 
## description per the defined response
class OpenAI:
    """OpenAI API client."""

    def __init__(self):
        """Initialize OpenAI client."""
        self.model = os.environ.get("OPENAI_MODEL")
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.client = openai
        self.client.api_key = self.api_key

    def chat_completion_create(self, messages: list, **kwargs) -> dict:
        """
        Create chat completion.

        Parameters
        ----------
        messages : list
            list of messages to send to OpenAI
        **kwargs : dict
            additional arguments to pass to the OpenAI API
            expected keys: functions: array, function_call: dict

        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
                **kwargs
            )

            if kwargs.get("functions", None):
                message = response.choices[0].message
                data_string = message.function_call.arguments
                data = json.loads(data_string)
                return data

            content = json.loads(response.choices[0].message.content)
            return content
        except Exception as e:
            object = {
                "error": "Error occurred while fetching from OpenAI API",
                "error_message": str(e),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "messages": messages,
            }

        return object

    def get_pokemon(self, name: str = "Pikachu") -> Dict[Any, Any]:
        """Get a Pokemon description from OpenAI."""

        init = "Provide a holisitic description of {name}, a Pokemon.".format(
            name=name
        )

        messages = [
            {"role": "system", "content": "You are a Pokemon master and you've caught them all. You now share your knowledge with the world."},
            {
                "role": "user",
                "content": init,
            },
        ]

        functions = [
            {
                "name": "get_answer_for_user_query",
                "description": "Get user answer in series of steps",
                "parameters": PokemonOpenAIContract.model_json_schema(),
            }
        ]

        function_call = {"name": "get_answer_for_user_query"}

        return self.chat_completion_create(
            messages, **{"functions": functions, "function_call": function_call}
        )

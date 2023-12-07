import os
import cohere
import openai
from dotenv import load_dotenv
from fastapi import FastAPI

#Loading environment variables
load_dotenv()

app = FastAPI()

#Initializing the Cohere client
co = cohere.Client(os.getenv('COHERE_API_KEY'))
openai.api_key = os.getenv('OPENAI_API_KEY')

def generate_prompt(description):
    prompt = f'''Please predict dog's breed in the description.

    Description: This dog is a medium-sized breed of dog that was developed in the United States. The breed is often used as a working dog on farms and ranches, and is also a popular companion animal.
Breed: Australian Shepher
--
Description: This dog is a cheerful, high-spirited breed that is well suited for a variety of activities, including hunting, agility, and obedience. They are very intelligent and easily trained, and are excellent companions for active families. They are also known for their friendly dispositions and love of children.
Breed: Brittany
--
Description: They are large, muscular dogs with a short, thick coat. They have a wide head and a short, square muzzle. Their eyes are small and their ears are floppy. They have a thick neck and a strong, muscular body. Their tail is short and their legs are muscular.
Breed: American Bulldog
--
Description: This dog is a medium-sized breed of dog that is native to Poland. They are intelligent and active dogs that are known for their herding abilities. They have a thick, shaggy coat that can be either black, white, or brindle in color. They are loyal and protective dogs that make great family pets.
Breed: Polish Lowland Sheepdog
--
Description: They are large, working breed of dog known for its obedience, loyalty, and intelligence. They are strong, agile dogs with a well-proportioned build. They have a long head, erect ears, and a long, thick coat. The breed is intelligent and easily trained, making them excellent working dogs. They are used in a variety of roles, including law enforcement, search and rescue, and as assistance dogs.
Breed: German Shepherd Dog
--
Description: They are a medium-sized breed of dog, originally bred in England as gun dogs. They are bred to be lovable companions and are often considered one of the best breeds for families. They are relatively easy to train and are known for being intelligent and eager to please. Cockers have a reputation for being one of the most affectionate dog breeds, and are also known for being good with children.
Breed: Cocker Spaniel
--
Description: {description}
Breed:
'''
    return prompt

@app.get("/")
def breed_prediction(description: str):
    prompt = generate_prompt(description)

    response = co.generate(model='xlarge', prompt=prompt, max_token=5, temperature=0.75, stop_sequences=['--'],)

    breed = response.generations[0].text.strip()

    img = openai.Image.create(prompt=f'Image of {breed}', n=1, size = '512x512')

    img = img['data'][0]['url']

    return {
        'breed': breed,
        'img': img
    }


import requests
# 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
class ExampleGenerator:
    def __init__(self, api_url="https://api-inference.huggingface.co/models/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5", api_token="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", device='cpu'):
        self.api_url = api_url
        self.api_token = api_token
        self.device = device

    def generate_examples(self, word, definition, num_examples=10):
        payload = {
            "inputs": f"Give {num_examples+1} examples using the word '{word}' which means '{definition}'"
        }
        query = payload["inputs"]
        headers = {"Authorization": f"Bearer {self.api_token}"}

        output_text = ""
        i = 0
        while i < num_examples:
            response = requests.post(self.api_url, headers=headers, json=payload)
            output = response.json()
            if len(output) > 1 and output['error']:
                break
            generated_text = output[0]["generated_text"]
            output_text = generated_text
            payload["inputs"] = generated_text
            i += 1

        examples_list = []
        example_lines = output_text.split("\n")

        for line in example_lines:
            if line.startswith("Give"):
                continue
            if line.strip() == "":
                continue
            if len(examples_list) == num_examples:
                break
            examples_list.append(line.split('.')[1].strip())

        return set(examples_list)
        # return output_text

# # Example usage
# api_url = "https://api-inference.huggingface.co/models/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"
# api_token = "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"
# device = 'cpu'  # Assuming API inference is done on CPU

# generator = ExampleGenerator(api_url, api_token, device)
# word = "bat"
# definition = "hit or try to hit the ball"

# examples = generator.generate_examples(word, definition, 15)
# print(examples)

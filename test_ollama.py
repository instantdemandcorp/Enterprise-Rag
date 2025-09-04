import requests
import json

def generate_response_stream(model: str, prompt: str, host: str = "http://localhost:11434") -> str:
    """
    Sends a prompt to the Ollama API and streams the response safely.

    Args:
        model (str): The name of the model to use (e.g., "llama3.1").
        prompt (str): The input prompt to send.
        host (str): The base URL of the Ollama API.

    Returns:
        str: The full generated response text.
    """
    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt
    }

    try:
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()

        full_output = ""
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode("utf-8"))
                    full_output += chunk.get("response", "")
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
        return full_output

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama API: {e}")
        return ""

# Example usage
if __name__ == "__main__":
    result = generate_response_stream("llama3.1", "hi what is planet")
    print(result)
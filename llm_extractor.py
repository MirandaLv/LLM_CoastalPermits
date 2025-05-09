
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import textwrap


class HuggingFacePermitExtractor:
    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        load_in_4bit: bool = True,
        torch_dtype=torch.bfloat16,
        default_generate_kwargs: dict = None,
        max_input_tokens: int = 3500,  # stay under model limit
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            load_in_4bit=load_in_4bit,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        self.default_generate_kwargs = default_generate_kwargs or {
            "do_sample": False,
            "max_new_tokens": 1000,
            "temperature": 0.3,
        }
        self.max_input_tokens = max_input_tokens

    def _truncate_input(self, text: str) -> str:
        tokens = self.tokenizer.encode(text, truncation=True, max_length=self.max_input_tokens)
        return self.tokenizer.decode(tokens)

    def build_prompt(self, unstructured_text: str) -> list:
        truncated_text = self._truncate_input(unstructured_text)
        prompt = (
            "You are an expert assistant that extracts structured information from permit applications. "
            "Given the following text from a coastal structure permit application, extract the relevant details in JSON format, "
            "with keys such as: applicant_name, project_title, project_location, permit_id (if any), date_received, "
            "proposed_structure, purpose\n\n"
            f"Permit Text:\n{truncated_text}\n\n"
            "JSON Output:"
        ) #and any other important fields.
        return [{"role": "user", "content": prompt}]

    def extract_structured_data(self, permit_text: str, generate_kwargs: dict = None) -> str:
        messages = self.build_prompt(permit_text)
        model_inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        input_length = model_inputs.shape[1]

        gen_args = self.default_generate_kwargs.copy()
        if generate_kwargs:
            gen_args.update(generate_kwargs)

        with torch.no_grad():
            generated_ids = self.model.generate(model_inputs, **gen_args)

        output_text = self.tokenizer.decode(
            generated_ids[:, input_length:].squeeze(), skip_special_tokens=True
        )
        return output_text


# Example usage
if __name__ == "__main__":
    llm_extractor = HuggingFacePermitExtractor(
        model_name="HuggingFaceH4/zephyr-7b-alpha",
        default_generate_kwargs={
            "do_sample": False,
            "max_new_tokens": 1000,
            "temperature": 0.2,
        },
    )

    # Replace with your real permit application content
    raw_text = """
    Applicant: John Doe
    Location: 1025 Coastal Road, Virginia Beach, VA
    Proposed Project: Construction of a riprap revetment along 80 feet of shoreline
    Date Submitted: February 12, 2024
    Purpose: Shoreline erosion protection
    Permit ID: VMRC-2024-0912
    """

    output = llm_extractor.extract_structured_data(raw_text)
    print(output)


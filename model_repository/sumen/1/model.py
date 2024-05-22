import io
import torch
import json
import argparse
import numpy as np
from PIL import Image
from transformers import (
    AutoProcessor,
    VisionEncoderDecoderModel
)
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    
    def initialize(self, args):
        
        if torch.cuda.is_available():    
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Init model
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "hoang-quoc-trung/sumen-base"
        ).to(self.device)
        # Load processor
        self.processor = AutoProcessor.from_pretrained("hoang-quoc-trung/sumen-base")

        task_prompt = self.processor.tokenizer.bos_token
        self.decoder_input_ids = self.processor.tokenizer(
            task_prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids

        model_config = json.loads(args['model_config'])
        output0_config = pb_utils.get_output_config_by_name(model_config, "latex_string")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )


    def execute(self, requests):
        responses = []
        for request in requests:

            input_tensor = pb_utils.get_input_tensor_by_name(request, "input_image")
            image = input_tensor.as_numpy()
            image = Image.open(io.BytesIO(image.tobytes()))
            
            if not image.mode == "RGB":
                image = image.convert('RGB')

            pixel_values = self.processor.image_processor(
                image,
                return_tensors="pt",
                data_format="channels_first",
            ).pixel_values
            
            # Generate LaTeX expression
            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values.to(self.device),
                    decoder_input_ids=self.decoder_input_ids.to(self.device),
                    max_length=self.model.decoder.config.max_length,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=4,
                    bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                    return_dict_in_generate=True,
                )
            sequence = self.processor.tokenizer.batch_decode(outputs.sequences)[0]
            sequence = sequence.replace(
                    self.processor.tokenizer.eos_token, ""
                ).replace(
                    self.processor.tokenizer.pad_token, ""
                ).replace(self.processor.tokenizer.bos_token,"")
            output_tensors = pb_utils.Tensor(
                'latex_string', np.array(sequence).astype(self.output0_dtype)
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensors]
            )
            responses.append(inference_response)
        return responses
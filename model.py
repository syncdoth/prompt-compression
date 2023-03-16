from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

from utils import freeze_net


def load_transformer_LM_tokenizer(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if 't5' in model_name_or_path or 't0' in model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        # open-ended generation
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        model.config.bos_token_id = model.config.eos_token_id
    model.eval()
    freeze_net(model)

    return model, tokenizer

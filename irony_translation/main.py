from SarcasmTranslator import T5SarcasmTranslator
import json

if __name__ == "__main__":
    
    #Change medium to small or large if you want to test on those datasets. Make sure to change the dataset name and the output directory accordingly.
    ROOT_DIR = "./from_medium_dataset/"
    DATASET = "dataset_medium.json"  
    RESULTS_DIR = "./output_medium_results/" 
    TEST_DATASET = "./test_dataset.json" 
    

    models = [
        #{"model_name": "t5-small",           "output_dir": "./t5_small"},
        #{"model_name": "t5-base",            "output_dir": "./t5_base"},
        #{"model_name": "google-t5/t5-small", "output_dir": "./google_t5_small"},
        #{"model_name": "google-t5/t5-base",  "output_dir": "./google_t5_base"},
        #{"model_name": "google/mt5-small",   "output_dir": "./google_mt5_small"},
        #{"model_name": "google/mt5-base",    "output_dir": "./google_mt5_base"},
        {"model_name": "csebuetnlp/mT5_multilingual_XLSum",    "output_dir": "./mT5_multilingual_XLSum"},
        {"model_name": "facebook/nllb-200-distilled-600M",    "output_dir": "./nllb_200_distilled_600M"},
    ]

    sarcastic_sentence = []
    
    with open(TEST_DATASET) as f:
            sarcastic_sentence = json.load(f)
            sarcastic_sentence = sarcastic_sentence["sentences"]

    for m in models:
        translator = T5SarcasmTranslator(model_name=m["model_name"])    
        translator.load_model(f"{ROOT_DIR}{m['output_dir']}", dataset=DATASET)
        safe_name = m["model_name"].replace("/", "_")  # google-t5/t5-small -> google-t5_t5-small
        with open(f"{RESULTS_DIR}{safe_name}_results.txt", "w") as f:
            pass

        for sentence in sarcastic_sentence:
            output = translator.generate(sentence)
            with open(f"{RESULTS_DIR}{safe_name}_results.txt", "a") as f:
                f.write(f"[{m['model_name']}] INPUT: {sentence}\n")
                f.write(f"[{m['model_name']}] OUTPUT: {output}\n\n")
            print(f"[{m['model_name']}] OUTPUT: {output}")
from LLMSarcasmTranslator import LLMSarcasmTranslator
import json

if __name__ == "__main__":
    
    TEST_DATASET = "./test_dataset.json" 
    RESULTS_FILE = "./llm_results/"
    
    sarcastic_sentence = []
    
    with open(TEST_DATASET) as f:
            sarcastic_sentence = json.load(f)
            sarcastic_sentence = sarcastic_sentence["sentences"]
                           
    translator = LLMSarcasmTranslator(api_key="YOUR_API_KEY")
    
    print("Zero-shot results:")
    for sentence in sarcastic_sentence:
        output = translator.zero_shot(sentence)        
        with open(f"{RESULTS_FILE}zero_shot_results.txt", "a") as f:
            f.write(f" INPUT: {sentence}\n")
            f.write(f" OUTPUT: {output}\n\n")    
            print(f" INPUT: {sentence}")
            print(f" OUTPUT: {output}") 
                        
    print("Few-shot results:")
    for sentence in sarcastic_sentence:        
        output = translator.few_shot(sentence)
        with open(f"{RESULTS_FILE}few_shot_results.txt", "a") as f:
            f.write(f" INPUT: {sentence}\n")
            f.write(f" OUTPUT: {output}\n\n")    
            print(f" INPUT: {sentence}")
            print(f" OUTPUT: {output}")
            
           
            
    


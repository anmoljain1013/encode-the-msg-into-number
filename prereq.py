import tensorflow as tf
from transformers import BertTokenizer
from transformers import DistilBertTokenizer, TFDistilBertModel
from transformers import  DistilBertConfig
s = "very long corpus..."
words = s.split(" ")  # Split over space
vocabulary = dict(enumerate(set(words)))  # Map storing the word to it's corresponding id

print(vocabulary)
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased") 
print(bert_tokenizer.cls_token)
enc = bert_tokenizer.encode("Hi, I am anmol jain !")
print(enc,5675)

print(bert_tokenizer.decode(enc),822828)
print(bert_tokenizer.decode([8104]),6785) # no. kis alphabet nu represent krra pta lgjata 

distil_bert = 'distilbert-base-uncased' # Name of the pretrained models

#DistilBERT 
tokenizer = DistilBertTokenizer.from_pretrained(distil_bert)
model = TFDistilBertModel.from_pretrained(distil_bert)
e = tokenizer.encode("Hello, my dog is cute")
print(e,65656454654654)

input = tf.constant(e)[None, :]  # Batch size 1 
print(input,543322)
print(type(input)) # shape: [1,8]

output = model(input)

print(type(output))
print(len(output))
print(output,543435343534)
print((output[0])[0,0,:])

config = DistilBertConfig.from_pretrained(distil_bert, output_hidden_states=True)
x = tokenizer.encode("Hello, my dog is cute")
print(x,143)
input = tf.constant(x)[None, :]  # Batch size 1 
print(input,111111)
model = TFDistilBertModel.from_pretrained(distil_bert, config=config)
print(model,66666666)
print(model.config,90909090909)
output = model(input)
print(len(output)) 
output[0].shape
output[1][0].shape
print('hhfbjdfkfhkjdfuihhjjfjurjkjk')
print(type(output[1]))
print ("jhhfhhgyg g ugusdguggffuyug")
print(len(output[1])) # 7 Why?
print(output[1][6],888888888) 

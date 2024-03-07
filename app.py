from flask import Flask, request, render_template
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = Flask(__name__)
app.jinja_env.globals.update(zip=zip)


model = T5ForConditionalGeneration.from_pretrained("Unbabel/gec-t5_small")
tokenizer = T5Tokenizer.from_pretrained('t5-small')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    text = request.form['text']
    tokenized = tokenizer('gec: ' + text, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
    translation = tokenizer.decode(
    model.generate(
        input_ids = tokenized.input_ids,
        attention_mask = tokenized.attention_mask, 
        max_length=128,
        num_beams=5,
        early_stopping=True,
    )[0],
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=True
)
    return render_template('index.html', text=text, translation=translation)

if __name__ == '__main__':
    app.run(debug=True)

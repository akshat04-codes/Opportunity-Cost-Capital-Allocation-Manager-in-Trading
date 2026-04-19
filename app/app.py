import gradio as gr

def predict(price):
    return f"Predicted value: {price}"

demo = gr.Interface(fn=predict, inputs="text", outputs="text")
demo.launch()
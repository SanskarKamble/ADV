import google.generativeai as genai
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import sklearn

if 'messages' not in st.session_state:
    st.session_state.messages = []


if 'first_run' not in st.session_state:
    st.session_state.first_run = False


def prepend_prompt_format(prompt, data):
    return f"Your task is to give answer in two sections, First section will begin with #ANSWER# and following it would be one to two line answer. Second section will be code (if applicable) and it will begin with #CODE# followed by python code which will always be related to matplotlib visualization. Also assume that data is present in the 'data' variable. Don't modify data variable directly. Also remember that the code you generate will be given in the exec() function of python. Don't mention anything about code the user should not know that there is code. If code is not required at all or no visualization is asked, then create empty section #CODE#nocode. Generate Python code to visualize this dataset: {data.head().to_string()}.\nQuery: {prompt}"

def show_user_message(message):
    st.chat_message("user").write(message['parts'][0])


def exec_chart_code(code, data):
    if not code:
        return None

    try:
        exec_locals = {}
        exec(code, {"plt": plt, "sns": sns, "pd": pd, 'sklearn': sklearn, "data": data}, exec_locals)
        plot_buffer = io.BytesIO()
        plt.savefig(plot_buffer, format='png')
        plt.close()
        return plot_buffer
    except Exception as e:
        st.error(f"Error in executing the generated code: {str(e)}")
        return None



def show_assistant_message(message):
    answer = message.parts[0].text.split("#ANSWER#")[1].split("#CODE#")[0].strip()
    code = message.parts[0].text.split("#CODE#")[1].strip()
    if code.startswith('```python'):
        code = code[9:-3]
    if code.startswith('nocode'):
        code = ''

    st.chat_message("assistant").write(answer)
    if code:
        print(code)
        st.code(code, language='python')
        plot_buffer = exec_chart_code(code, data)
        if plot_buffer:
            st.image(plot_buffer)
        

genai.configure(api_key="AIzaSyDJvr5jvwg4ylUPU418eHU5Litv9RVAzeY")
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

st.title("Chat with Your Dataset for Data Visualization")

uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    print(data.head().to_string())
    st.write("Here's a preview of your dataset:", data.head())

    st.write("### Chat Section")

    
    # history
    for message in st.session_state.messages:
        if hasattr(message, 'role') and message.role == 'model':
            show_assistant_message(message)
        elif message['role'] == 'user':
            show_user_message(message)

    if not st.session_state.first_run:
        st.session_state.first_run = True
        first_message = [
            {
                'role': 'user',
                'parts': [prepend_prompt_format("Tell me few lines about the dataset and then Show 4 visualization in a single plot that will be the most relevant for this dataset (let them all be of different chart type like scatterplot, histogram, boxplot, etc)", data)]
            }]
        
        response = model.generate_content(
            first_message
        )
        st.session_state.messages.append(response.candidates[0].content)    
        show_assistant_message(st.session_state.messages[-1])
        

    prompt = st.chat_input("Describe the visualization you want (e.g., 'scatter plot of age vs. income')")

  
    if prompt:

        st.session_state.messages.append(
            {
                'role': "user",
                'parts': [prompt]
            }
        )

        show_user_message(st.session_state.messages[-1])

        conversation = []

        for message in st.session_state.messages:
            if hasattr(message, 'role') and message.role == 'model':
                conversation.append(message)
            elif message['role'] == 'user':
                conversation.append({
                    'role': 'user',
                    'parts': [prepend_prompt_format(message['parts'][0], data)]
                })


        response = model.generate_content(
            conversation
        )

        
        st.session_state.messages.append(response.candidates[0].content)

        show_assistant_message(st.session_state.messages[-1])
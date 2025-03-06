import streamlit as st
import pandas as pd
import os
from openai import OpenAI
import streamlit
import packaging.version
import re
import matplotlib.pyplot as plt
import io
import base64
import json  # Add this import for JSON handling



# Model mapping dictionary (pseudonames to actual model names)
MODEL_MAPPING = {
    "Shelter-Mini": "gpt-4o"
}

# Initialize session state for API key status
if "api_key_submitted" not in st.session_state:
    st.session_state.api_key_submitted = False

if "client" not in st.session_state:
    st.session_state.client = None

# API Key Form (shown before the main app)
if not st.session_state.api_key_submitted:
    st.markdown("<h1 style='text-align: center;'>Welcome to Childrens Shelter AI</h1>", unsafe_allow_html=True)
    
    st_version = packaging.version.parse(streamlit.__version__)
    min_version_for_border = packaging.version.parse("1.22.0")

    # Display version info
    st.sidebar.markdown(f"<p style='font-size: 0.8em; color: #666;'>Streamlit version: {streamlit.__version__}</p>", unsafe_allow_html=True)

    # Use different form approaches based on version
    if st_version >= min_version_for_border:
        with st.form("api_key_form", border=False):
            st.markdown("<h3 style='text-align: center;'>Please enter your OpenAI API key to continue</h3>", unsafe_allow_html=True)
            
            # Check if API key is in secrets
            if "OPENAI_API_KEY" in st.secrets:
                api_key = st.secrets["OPENAI_API_KEY"]
                st.success("API good to go!")
            else:
                api_key = st.text_input("OpenAI API Key", type="password")
            
            # Submit button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                submitted = st.form_submit_button("Start Application", use_container_width=True)
    else:
        # Use the HTML/CSS approach for older versions
        with st.form("api_key_form"):
            st.markdown("<h3 style='text-align: center;'>Please enter your OpenAI API key to continue</h3>", unsafe_allow_html=True)
            
            # Check if API key is in secrets
            if "OPENAI_API_KEY" in st.secrets:
                api_key = st.secrets["OPENAI_API_KEY"]
                st.success("API good to go!")
            else:
                api_key = st.text_input("OpenAI API Key", type="password")
            
            # Submit button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                submitted = st.form_submit_button("Start Application", use_container_width=True)
    
    if submitted:
        if api_key:
            try:
                # Test the API key with a simple request
                client = OpenAI(api_key=api_key)
                st.session_state.client = client
                st.session_state.api_key_submitted = True
                st.rerun()  # Rerun the app to show the main interface
            except Exception as e:
                st.error(f"Error with API key: {str(e)}")
        else:
            st.error("Please enter an API key to continue")
    
    # Stop execution here if API key not submitted
    st.stop()

# Main application (only shown after API key is submitted)
st.markdown("<h1 style='text-align: center;'>Childrens Shelter AI</h1>", unsafe_allow_html=True)

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "Shelter-Mini"
    st.session_state["actual_model"] = MODEL_MAPPING["Shelter-Mini"]

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize spreadsheet data
if "spreadsheet_data" not in st.session_state:
    st.session_state.spreadsheet_data = None
    st.session_state.df = None

# Sidebar for configuration and file upload
with st.sidebar:
    st.markdown("<h3>Data Source</h3>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload your spreadsheet", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            
            if file_extension == '.csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['.xls', '.xlsx']:
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please use CSV or Excel files.")
                df = None
                
            if df is not None:
                st.session_state.spreadsheet_data = df.to_csv(index=False)
                st.session_state.df = df
                st.success(f"Successfully loaded {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error reading the spreadsheet: {e}")
    
    # Model information
    st.markdown("<h3>Model Settings</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.9em;'>Using Shelter-Mini: Our most powerful model, best for complex analysis and detailed insights.</p>", unsafe_allow_html=True)
    
    # Add option to reset API key
    st.markdown("<h3>API Settings</h3>", unsafe_allow_html=True)
    if st.button("Change API Key", use_container_width=True):
        st.session_state.api_key_submitted = False
        st.rerun()

# Display data preview if available
if st.session_state.df is not None:
    with st.expander("📊 Data Preview", expanded=False):
        st.dataframe(st.session_state.df, use_container_width=True)
        st.markdown(f"<p><strong>{len(st.session_state.df)}</strong> rows, <strong>{len(st.session_state.df.columns)}</strong> columns</p>", unsafe_allow_html=True)

# Function to extract and execute Python chart code
def extract_and_render_chart(response_text):
    # Look for Python code blocks that might contain chart generation
    code_blocks = re.findall(r'```python\s*(.*?)\s*```', response_text, re.DOTALL)
    
    charts_rendered = False
    
    for code in code_blocks:
        # Check if this code block contains chart-related imports or functions
        chart_indicators = [
            'matplotlib', 'pyplot', 'plt.', 'seaborn', 'sns.', 
            'plotly', 'px.', 'go.', 'chart', 'plot(', 'figure', 
            'bar(', 'line(', 'scatter(', 'hist('
        ]
        
        if any(indicator in code for indicator in chart_indicators):
            try:
                # Create a safe locals dictionary with pandas and the dataframe
                locals_dict = {
                    'pd': pd,
                    'plt': plt,
                    'df': st.session_state.df,
                    'data': st.session_state.df,  # Alternative name
                    'matplotlib': __import__('matplotlib'),
                    'np': __import__('numpy'),
                    'StringIO': __import__('io').StringIO,  # Add StringIO for examples with embedded data
                }
                
                # Try to execute the code
                exec(code, globals(), locals_dict)
                
                # If matplotlib figure was created, display it
                if 'plt' in locals_dict and plt.get_fignums():
                    st.pyplot(plt.gcf())
                    plt.close()
                    charts_rendered = True
                
                # If a figure was returned and stored in a variable, try to display it
                for var_name, var_value in locals_dict.items():
                    if var_name not in ['pd', 'plt', 'df', 'data', 'matplotlib', 'np', 'StringIO']:
                        if hasattr(var_value, 'figure') or str(type(var_value)).find('Figure') != -1:
                            st.pyplot(var_value)
                            charts_rendered = True
            except Exception as e:
                st.warning(f"Could not render chart from code: {str(e)}")
    
    return charts_rendered

# Function to render chart from JSON data
def render_chart_from_json(chart_data):
    try:
        chart_type = chart_data.get('chart_type', '').lower()
        title = chart_data.get('title', 'Chart')
        x_label = chart_data.get('x_label', '')
        y_label = chart_data.get('y_label', '')
        data = chart_data.get('data', {})
        
        # Create figure with more height for legend
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if chart_type == 'line':
            x = data.get('x', [])
            y = data.get('y', [])
            labels = data.get('labels', [])
            
            # Check if y is 2D array (multiple lines)
            if isinstance(y, list) and len(y) > 0 and isinstance(y[0], list):
                # Plot each line with its corresponding label
                for i, line_data in enumerate(y):
                    # Use provided label or fallback to index
                    label = labels[i] if i < len(labels) else f"Patient {i+1}"
                    
                    # Convert line_data to float to handle any string numbers
                    line_data = [float(val) if val is not None else 0 for val in line_data]
                    
                    # Plot with distinct colors and markers
                    ax.plot(x, line_data, 
                           label=label, 
                           marker='o', 
                           markersize=6, 
                           linewidth=2,
                           markeredgecolor='white',
                           markeredgewidth=1)
                
                # Rotate x-axis labels for better readability
                plt.xticks(rotation=45, ha='right')
                
                # Adjust layout to prevent label cutoff
                plt.tight_layout()
                
                # Add legend outside of plot area with larger font
                legend = ax.legend(bbox_to_anchor=(1.05, 1), 
                                 loc='upper left', 
                                 borderaxespad=0.,
                                 fontsize=10,
                                 title="Patients",
                                 title_fontsize=12)
                
                # Add some padding to the right for the legend
                plt.subplots_adjust(right=0.85)
            else:
                # Single line plot
                ax.plot(x, y, marker='o')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
        
        elif chart_type == 'bar':
            x = data.get('x', [])
            y = data.get('y', [])
            ax.bar(x, y)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
        elif chart_type == 'scatter':
            x = data.get('x', [])
            y = data.get('y', [])
            ax.scatter(x, y)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
        elif chart_type == 'pie':
            labels = data.get('labels', [])
            values = data.get('values', [])
            ax.pie(values, labels=labels, autopct='%1.1f%%')
            ax.axis('equal')
        elif chart_type == 'histogram':
            values = data.get('values', [])
            bins = data.get('bins', 10)
            ax.hist(values, bins=bins)
            plt.tight_layout()
        else:
            st.warning(f"Unsupported chart type: {chart_type}")
            return False
        
        ax.set_title(title, pad=20, fontsize=14)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        
        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set y-axis to start from 0
        if chart_type in ['line', 'bar', 'scatter']:
            ax.set_ylim(bottom=0)
            
        # Increase tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        st.pyplot(fig)
        plt.close(fig)
        return True
    except Exception as e:
        st.warning(f"Error rendering chart from JSON: {str(e)}")
        st.warning(f"Data received: {json.dumps(data, indent=2)}")  # Add this for debugging
        return False

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Check if this is a function call result with chart data
        if message["role"] == "assistant" and "function_call" in message:
            try:
                function_data = json.loads(message["function_call"])
                if function_data.get("name") == "generate_chart":
                    chart_data = function_data.get("arguments", {})
                    render_chart_from_json(chart_data)
            except:
                # Try to render any charts from code blocks
                if message["role"] == "assistant":
                    extract_and_render_chart(message["content"])

# Function to generate response from OpenAI
def generate_response(prompt):
    if not st.session_state.spreadsheet_data:
        return "Please upload a spreadsheet first."
    
    # Check if the prompt is asking for a chart
    chart_keywords = ["chart", "graph", "plot", "visualize", "visualization", "trend", "historical", "display"]
    is_chart_request = any(keyword in prompt.lower() for keyword in chart_keywords)
    
    try:
        # Define function for chart generation
        functions = [
            {
                "name": "generate_chart",
                "description": "Generate a chart based on the data analysis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chart_type": {
                            "type": "string",
                            "description": "The type of chart to generate (bar, line, scatter, pie, histogram)",
                            "enum": ["bar", "line", "scatter", "pie", "histogram"]
                        },
                        "title": {
                            "type": "string",
                            "description": "The title of the chart"
                        },
                        "x_label": {
                            "type": "string",
                            "description": "The label for the x-axis"
                        },
                        "y_label": {
                            "type": "string",
                            "description": "The label for the y-axis"
                        },
                        "data": {
                            "type": "object",
                            "description": "The data for the chart",
                            "properties": {
                                "x": {
                                    "type": "array",
                                    "description": "The x values for the chart (for bar, line, scatter)",
                                    "items": {
                                        "type": ["string", "number"]
                                    }
                                },
                                "y": {
                                    "type": "array",
                                    "description": "The y values for the chart (for bar, line, scatter)",
                                    "items": {
                                        "type": "number"
                                    }
                                },
                                "labels": {
                                    "type": "array",
                                    "description": "The labels for the chart (for pie)",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "values": {
                                    "type": "array",
                                    "description": "The values for the chart (for pie, histogram)",
                                    "items": {
                                        "type": "number"
                                    }
                                },
                                "bins": {
                                    "type": "integer",
                                    "description": "The number of bins for a histogram"
                                }
                            }
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Explanation of what the chart shows and key insights"
                        }
                    },
                    "required": ["chart_type", "title", "data", "explanation"]
                }
            }
        ]
        
        # Adjust the system message based on whether a chart is requested
        system_content = "You are a data scientist that analyzes and works with provided spreadsheet data and provides a detailed report based on the query you are given. Respond briefly and concisely."
        
        if is_chart_request:
            system_content += """ If the user asks for a chart or visualization, use the generate_chart function to create the requested visualization with appropriate data from the spreadsheet. 
            IMPORTANT: When plotting patient data:
            1. Always use the patient names from the 'Name' column in the spreadsheet as labels
            2. Make sure to include these names in the 'labels' array of the chart data
            3. Format dates consistently and ensure the data is properly sorted chronologically
            4. The data structure should look like this:
            {
                "chart_type": "line",
                "title": "PHQ-9 Scores Over Time",
                "x_label": "Date",
                "y_label": "PHQ-9 Score",
                "data": {
                    "x": ["date1", "date2", ...],
                    "y": [[score1, score2, ...], [score1, score2, ...], ...],
                    "labels": ["Patient Name 1", "Patient Name 2", ...]
                }
            }
            """
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Here is the spreadsheet data:\n\n{st.session_state.spreadsheet_data}\n\nNow, {prompt}"}
        ]
        
        # Call the API with function calling enabled
        if is_chart_request:
            response = st.session_state.client.chat.completions.create(
                model=st.session_state["actual_model"],
                messages=messages,
                functions=functions,
                function_call="auto",
                temperature=0.7
            )
        else:
            response = st.session_state.client.chat.completions.create(
                model=st.session_state["actual_model"],
                messages=messages,
                temperature=0.7
            )
        
        message = response.choices[0].message
        
        # Check if the model wants to call a function
        if hasattr(message, 'function_call') and message.function_call:
            # Extract the function call
            function_call = message.function_call
            function_name = function_call.name
            function_args = json.loads(function_call.arguments)
            
            # Store both the text response and function call data
            result = {
                "content": message.content or "Here's the chart based on your request:",
                "function_call": json.dumps({
                    "name": function_name,
                    "arguments": function_args
                })
            }
            return result
        else:
            # Return just the content for regular responses
            return message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Accept user input
if prompt := st.chat_input("Ask a question about your data..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        response = generate_response(prompt)
        
        # Check if response is a dictionary with function call
        if isinstance(response, dict) and "function_call" in response:
            # Remove copy button and just display the text response
            message_placeholder.markdown(response["content"])
            
            # Process the function call
            try:
                function_data = json.loads(response["function_call"])
                if function_data["name"] == "generate_chart":
                    chart_data = function_data["arguments"]
                    if render_chart_from_json(chart_data):
                        st.success(f"Chart generated: {chart_data.get('title', 'Chart')}")
                        
                        # Display the explanation if available
                        explanation = chart_data.get("explanation", "")
                        if explanation:
                            st.markdown("### Analysis")
                            st.markdown(explanation)
            except Exception as e:
                st.error(f"Error processing function call: {str(e)}")
        else:
            # Remove copy button and just display the text response
            message_placeholder.markdown(response)
            
            # Try to render any charts from code blocks (fallback)
            extract_and_render_chart(response)
    
    # Add assistant response to chat history
    if isinstance(response, dict):
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response["content"],
            "function_call": response.get("function_call")
        })
    else:
        st.session_state.messages.append({"role": "assistant", "content": response})

# Display instructions if no data is loaded
if st.session_state.spreadsheet_data is None:
    st.markdown("""
    <div style="text-align: center;">
        <h3>Get Started</h3>
        <p>👈 Please upload a spreadsheet file in the sidebar to begin.</p>
        <p style="font-weight: bold;">Analyze your data with AI in seconds!</p>
    </div>
    """, unsafe_allow_html=True)

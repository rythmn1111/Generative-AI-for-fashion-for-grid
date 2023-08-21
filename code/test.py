from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.tools import DuckDuckGoSearchRun
import streamlit as st
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, LLMChain
from langchain.tools import DuckDuckGoSearchRun 
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re
import langchain
from langchain.tools import BaseTool
import os
import requests
from PIL import Image


st.set_page_config(page_title="Fashion Outfit Generator", page_icon="🔮")
# st.title("🔮 DaVinci Dresser")

openai_api_key = st.sidebar.text_input("Please insert your OpenAI API Key", type="password")

radio_btn = st.sidebar.radio(
    "Choose model",
    ('DaVinci Dresser', 'Curie Matcher'))
st.sidebar.info('Both our DaVinci Dresser and Curi Matcher are build using [langchain](https://www.langchain.com/) and are based on [MRKL](https://arxiv.org/pdf/2205.00445.pdf) and [ReAct](https://ai.googleblog.com/2022/11/react-synergizing-reasoning-and-acting.html). Both of our AI are capable of understanding all the fashion trends, knows what is trending on social media and also knows the user past purchase history/most viewed item and based on that assist user with there query.  ')
st.sidebar.write("Submission by BackendBoys")
if radio_btn == "DaVinci Dresser":
    st.title("🔮 DaVinci Dresser")
    with st.expander("DaVinci Dresser is a smart AI which can recommend fashion product for any given query"):
        image = Image.open('ss.png')
        st.image(image, caption='process')
    msgs = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(
        chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
    )
    if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
        msgs.clear()
        msgs.add_ai_message("Your personal fashion assistant is here to help you find the perfect outfit!")
        st.session_state.steps = {}

    avatars = {"human": "user", "ai": "assistant"}
    for idx, msg in enumerate(msgs.messages):
        with st.chat_message(avatars[msg.type]):
            # Render intermediate steps if any were saved
            for step in st.session_state.steps.get(str(idx), []):
                if step[0].tool == "_Exception":
                    continue
                with st.expander(f"✅ **{step[0].tool}**: {step[0].tool_input}"):
                    st.write(step[0].log)
                    st.write(f"**{step[1]}**")
            st.write(msg.content)

    if prompt121 := st.chat_input(placeholder="Show me something dark academia aesthetic"):
        st.chat_message("user").write(prompt121)

        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()   

    #setting up OpenAI api key
    os.environ["OPENAI_API_KEY"] = openai_api_key

    #setting up custom tool for product search
    class flipkart_search(BaseTool):
        name = "product_link_generator"
        description = "useful when you need to return link of a particular product. input should be a product name"
        
        def _run(self, item_name: str, num_items=3):
            api_base_url = "https://flipkart-scraper-api.dvishal485.workers.dev/search/"
            api_url = f"{api_base_url}{item_name}"

            response = requests.get(api_url)
            if response.status_code == 200:
                product_data = response.json()

                if 'result' in product_data:
                    result_list = product_data['result']
                    num_items = min(num_items, len(result_list))

                    product_links = [result_list[i].get('link') for i in range(num_items)]

                    return product_links
                else:
                    return None
            else:
                return None
                
    product_link_generator = flipkart_search()

    search = DuckDuckGoSearchRun()

    #defining the agent tools
    tools = [
        Tool(
            name = "Search",
            func=search.run,
            description="useful for when you need to answer questions about current events"
        ),
        Tool(
            name = "product_link_generator",
            func=product_link_generator.run,
            description="useful when you need to return link of a particular product. input should be a product name"
        )
    ]



    #setting up custom template for prompt
    template = """Answer the following question as best you can, so you are a fashion outfit and link generator bot for an Indian e-commerce company flipkart. Here is what you need to do when user give any query related to fashion first you need to search for what user wants and then search for what fashion items comes under that and at last you COMPULSORY need to search if those products are on flipkart and then finally you need to give the link at last using the tool product_link_generator. If user asks for some trend related outfit then first search what comes under that trend and then use product_link_generator to give link .REMEMBER two important things that you COMPULSORY need to provide the product link, second is the product link should be at last and if user asks something other than fashion query then dont answer that.  You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin! Remember to answer as a compasionate fashion outfit and accessories suggestion bot when giving your final answer.

    Question: {input}
    {agent_scratchpad}"""

    #Setting up a prompt template
    class CustomPromptTemplate(StringPromptTemplate):
        
        template: str
        
        tools: List[Tool]
        
        def format(self, **kwargs) -> str:
            
            
            intermediate_steps = kwargs.pop("intermediate_steps")
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += f"\nObservation: {observation}\nThought: "
            
            kwargs["agent_scratchpad"] = thoughts
            
            kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
            
            kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
            return self.template.format(**kwargs)
        
    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        
        input_variables=["input", "intermediate_steps"]
    )

    #setting up output parse

    class CustomOutputParser(AgentOutputParser):
        
        def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
            
            if "Final Answer:" in llm_output:
                return AgentFinish(
                    
                    return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                    log=llm_output,
                )
            # Parse out the action and action input
            regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
            match = re.search(regex, llm_output, re.DOTALL)
            if not match:
                raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            action = match.group(1).strip()
            action_input = match.group(2)
            # Return the action and action input
            return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
        
    output_parser = CustomOutputParser()
    llm = OpenAI(temperature=0,model_name='gpt-3.5-turbo')
   # llm = OpenAI(temperature=0)

    # LLM chain consisting of the LLM and a prompt
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    tool_names = [tool.name for tool in tools]

    agent = LLMSingleActionAgent(
        llm_chain=llm_chain, 
        output_parser=output_parser,
        stop=["\nObservation:"], 
        allowed_tools=tool_names
    )

    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, 
                                                        tools=tools,
                                                        return_intermediate_steps=True, 
                                                        verbose=True)

    #setting up the chat model
    #agent_executor.run(prompt121)

    with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = agent_executor(prompt121, callbacks=[st_cb])
            st.write(response["output"])
            st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]

else:
    st.title("🧬 Curie Matcher")
    with st.expander("Confused about what to wear with which outfit? Ask Curie Matcher."):
            st.write("Our AI powered fashion outfit generator  ")
    msgs = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(
        chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
    )
    if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
        msgs.clear()
        msgs.add_ai_message("Your personal fashion assistant is here to help you find the perfect outfit!")
        st.session_state.steps = {}
        
    avatars = {"human": "user", "ai": "assistant"}
    for idx, msg in enumerate(msgs.messages):
        with st.chat_message(avatars[msg.type]):
            # Render intermediate steps if any were saved
            for step in st.session_state.steps.get(str(idx), []):
                if step[0].tool == "_Exception":
                    continue
                with st.expander(f"✅ **{step[0].tool}**: {step[0].tool_input}"):
                    st.write(step[0].log)
                    st.write(f"**{step[1]}**")
            st.write(msg.content)

    if prompt121 := st.chat_input(placeholder="Show me something dark academia aesthetic"):
        st.chat_message("user").write(prompt121)

        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()   

    #setting up OpenAI api key
    os.environ["OPENAI_API_KEY"] = openai_api_key

    #setting up custom tool for product search
    class flipkart_search(BaseTool):
        name = "product_link_generator"
        description = "useful when you need to return link of a particular product. input should be a product name"
        
        def _run(self, item_name: str, num_items=3):
            api_base_url = "https://flipkart-scraper-api.dvishal485.workers.dev/search/"
            api_url = f"{api_base_url}{item_name}"

            response = requests.get(api_url)
            if response.status_code == 200:
                product_data = response.json()

                if 'result' in product_data:
                    result_list = product_data['result']
                    num_items = min(num_items, len(result_list))

                    product_links = [result_list[i].get('link') for i in range(num_items)]

                    return product_links
                else:
                    return None
            else:
                return None
                
    product_link_generator = flipkart_search()

    search = DuckDuckGoSearchRun()

    #defining the agent tools
    tools = [
        Tool(
            name = "Search",
            func=search.run,
            description="useful for when you need to answer questions about current events"
        )
    ]



    #setting up custom template for prompt
    template = """Answer the following question as best you can, so you are a fashion outfit generator to whom user can ask what they can wear with a particular cloth or any other fashion accessories, basically you need to find them fashion items which goes with there cloths or they can also ask you some particular trend and you need to tell them what kind of outfit they can wear for that trend.   You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin! Remember to answer as a compasionate fashion outfit and accessories suggestion bot when giving your final answer.

    Question: {input}
    {agent_scratchpad}"""

    #Setting up a prompt template
    class CustomPromptTemplate(StringPromptTemplate):
        
        template: str
        
        tools: List[Tool]
        
        def format(self, **kwargs) -> str:
            
            
            intermediate_steps = kwargs.pop("intermediate_steps")
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += f"\nObservation: {observation}\nThought: "
            
            kwargs["agent_scratchpad"] = thoughts
            
            kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
            
            kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
            return self.template.format(**kwargs)
        
    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        
        input_variables=["input", "intermediate_steps"]
    )

    #setting up output parse

    class CustomOutputParser(AgentOutputParser):
        
        def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
            
            if "Final Answer:" in llm_output:
                return AgentFinish(
                    
                    return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                    log=llm_output,
                )
            # Parse out the action and action input
            regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
            match = re.search(regex, llm_output, re.DOTALL)
            if not match:
                raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            action = match.group(1).strip()
            action_input = match.group(2)
            # Return the action and action input
            return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
        
    output_parser = CustomOutputParser()
    llm = OpenAI(temperature=0)

    # LLM chain consisting of the LLM and a prompt
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    tool_names = [tool.name for tool in tools]

    agent = LLMSingleActionAgent(
        llm_chain=llm_chain, 
        output_parser=output_parser,
        stop=["\nObservation:"], 
        allowed_tools=tool_names
    )

    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, 
                                                        tools=tools,
                                                        return_intermediate_steps=True, 
                                                        verbose=True)

    #setting up the chat model

    with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = agent_executor(prompt121, callbacks=[st_cb])
            st.write(response["output"])
            st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]

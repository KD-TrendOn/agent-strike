from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from typing import Literal
import json

load_dotenv()

ROOT_FOLDER = input("Введите абсолютный путь к корневому каталогу проекта:\n")
PROGRAMMING_LANGUAGE = input("Введите язык программирования или фреймворк на котором вы разрабатываете проект:\n")


OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY")
os.environ['LANGCHAIN_PROJECT'] = "Agent Strike"
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")

llm = ChatOpenAI(
        temperature=0,
        model="openai/gpt-4o",#anthropic/claude-3.5-sonnet
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
    )

class LanguageSchema(BaseModel):
    language:Literal['python', 'other'] = Field(description="Язык программирования либо python либо other. Поле служит для отсеивания запросов.")


def get_language(programming_language:str) -> str:
    if programming_language == 'python':
        return 'python'
    output_parser = PydanticOutputParser(pydantic_object=LanguageSchema)
    format_instructions = output_parser.get_format_instructions()
    prompt_template = """Ты эксперт которому нужно извлечь информацию из текста.
    Определи, введенный пользователем текст это python или other.
    Верни только python или other в нужном формате:

    {format_instructions}

    Текст сообщения:
    {programming_language}
    """
    prompt = PromptTemplate.from_template(template=prompt_template)
    chain = prompt | llm | output_parser
    return chain.invoke({'format_instructions':format_instructions, 'programming_language':programming_language}).dict()['language']

def get_prefix(language:Literal['python', 'other']='python', other_option:str=PROGRAMMING_LANGUAGE)->str:
    if language == 'python':
        prefix = """
        Ты помощник по программированию на Python. Ты умеешь писать тесты, проводить ревью кода, писать скрипты не только для питона но и для консоли, используя bat или bash файлы.
        Ты также можешь отвечать на вопросы и использовать инструменты. Ты агент, который может использовать инструменты, и твоя задача в зависимости от запроса
        подобрать стратегию по которой ты выполнишь задание. Сначала можешь использовать поиск в интернете только если нужно. Также сначала посмотри структуру файлов в папке перед записью и прочтением файлов. Далее ты можешь прочитать файлы которые тебе нужны
        Далее ты можешь либо составить отчет, выполнить ревью, ответить на вопрос пользователя по задаче либо создать или изменить документ в папке.
        Если запрос требует изменить что то в коде, сначала посмотри список файлов в папке, найди подходящий, прочти содержимое, затем запиши в него новое содержимое в зависимости от задачи.
        Если нужно написать тесты для какого то модуля, прочти файлы этого модуля и в папке tests создай файлы в которых с помощью pytests и unittests ты напишешь тесты.
        """
    else:
        prefix = f"""
        Ты помощник по программированию для на {other_option}. Ты умеешь писать тесты, проводить ревью кода, писать скрипты как и на данном языке, так и для консоли, используя bat или bash файлы.
        Ты также можешь отвечать на вопросы и использовать инструменты. Ты агент, который может использовать инструменты и твоя задача в зависимости от запроса
        подобрать стратегию по которой ты выполнишь задание. Сначала можешь использовать поиск в интернете только если нужно. Далее ты можешь прочитать файлы которые тебе нужны
        Далее ты можешь либо составить отчет, выполнить ревью, ответить на вопрос пользователя по задаче либо создать или изменить документ или скрипт в папке.
        Если запрос требует изменить что то в коде, сначала посмотри список файлов в папке, найди подходящий, прочти содержимое, затем запиши в него новое содержимое в зависимости от задачи.
        """
    return prefix
#Tools
#Поиск в интернете

tavily_tool = TavilySearchResults(max_results=5)

#Прочитать список документов в папке

@tool
def list_files() -> str:
    """Инструмент который возвращает список всех файлов в рабочей папке пользователя.

    Returns:
        str: Строка, в которой перечислены файлы которые ты можешь прочитать или редактировать.
    """
    print('list_files')
    result = ""
    for dirname, dirnames, filenames in os.walk(ROOT_FOLDER):
    # print path to all filenames.
        for filename in filenames:
            result += str(os.path.join(dirname, filename))[len(ROOT_FOLDER)+1:]
            result += '\n'
        # Advanced usage:
        # editing the 'dirnames' list will stop os.walk() from recursing into there.
        if '.git' in dirnames:
            # don't go into any .git directories.
            dirnames.remove('.git')
        if 'env' in dirnames:
            dirnames.remove('env')
        if '__pycache__' in dirnames:
            dirnames.remove('__pycache__')
        if 'agent' in dirnames:
            dirnames.remove('agent')
    return result

#Прочитать файл

class ReadInput(BaseModel):
    file_path: str = Field(description="Относительный путь к файлу в рабочей папке пользователя. Например folder/file.py или main.py. Полный список файлов и их местонахождения можно получить с помощью другого инструмента. Обязательно сначала удостоверься что файл существует.")

@tool("read-file", args_schema=ReadInput)
def read_file(file_path:str) -> str:
    """Инструмент для получения содержимого файла

    Args:
        file_path (str): Относительный путь к файлу в рабочей папке пользователя. Например folder/file.py или main.py. Полный список файлов и их местонахождения можно получить с помощью другого инструмента. Обязательно сначала удостоверься что файл существует.
    Returns:
        str: Содержимое файла либо сообщение что файла не существует.
    """
    print('read_file')
    if not os.path.exists(os.path.join(ROOT_FOLDER, file_path)):
        return "Файла не существует"
    with open(os.path.join(ROOT_FOLDER, file_path), 'r', encoding='utf-8') as f:
        return '\n'.join(f.readlines())

#Инструмент перезаписи

class WriteInput(BaseModel):
    file_path: str = Field(description="Относительный путь к файлу в рабочей папке пользователя. Например folder/file.py или main.py. Если файла не существует, для записи создастся новый файл с нужным именем. Если файл существует ты перезапишешь его содержимое, так что если нужно воспользуйся другим инструментом чтобы прочесть его содержимое. Полный список файлов и их местонахождения можно получить с помощью другого инструмента.")
    content: str = Field(description="Должно быть контентом для перезаписи или записи файла. Многострочный программный код или другое наполнение.")

def write_content(full_path:str, content:str) -> str:
    try:
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except:
        raise ValueError
    else:
        return "Positive"

@tool('write-file', args_schema=WriteInput)
def write_file(file_path:str, content:str) -> str:
    """Инструмент для создания и записи либо перезаписи файла. Если целью является изменение фрагмента файла, прочтите его содержимое перед перезаписью чтобы правильно перенести прошлые строчки и добавить свои.

    Args:
        file_path (str): Относительный путь к файлу в рабочей папке пользователя. Например folder/file.py или main.py. Если файла не существует, для записи создастся новый файл с нужным именем. Если файл существует ты перезапишешь его содержимое, так что если нужно воспользуйся другим инструментом чтобы прочесть его содержимое. Полный список файлов и их местонахождения можно получить с помощью другого инструмента.
        content (str): Должно быть контентом для перезаписи или записи файла. Многострочный программный код или другое наполнение.

    Returns:
        str: Статус запроса (Writed | Rewrited | Error | User reject)
    """
    print('write_file')
    user_input = int(input(f"Подтвердить запись в файл {file_path} контента:\n {content}\n 0 или 1"))
    if not user_input:
        return "User rejected request, try to ask what's wrong"
    if os.path.exists(os.path.join(ROOT_FOLDER, file_path)):
        try:
            write_content(os.path.join(ROOT_FOLDER, file_path), content)
        except:
            return 'Error'
        else:
            return 'Rewrited'
    else:
        if '/' in file_path:
            if not os.path.exists(os.path.join(ROOT_FOLDER, '/'.join(file_path.split('/')[:-1]))):
                os.makedirs(os.path.join(ROOT_FOLDER, '/'.join(file_path.split('/')[:-1])))
        try:
            write_content(os.path.join(ROOT_FOLDER, file_path), content)
        except:
            return 'Error'
        else:
            return 'Writed'

tools = [tavily_tool, list_files, read_file, write_file]

llm_with_tools = llm.bind_tools(tools)

from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages
)
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate
MEMORY_KEY = "chat_history"
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template=get_prefix(get_language(PROGRAMMING_LANGUAGE)),
            ),
        ),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=['input'],
                template='{input}'
            )
        ),
        MessagesPlaceholder(variable_name='agent_scratchpad'),
    ]
)

from langchain_core.messages import AIMessage, HumanMessage

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def msg_from_list(chat_history:list, file_path):
    result = []
    for i in chat_history:
        result.append(str(i.content))
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(result, f)


def list_from_msg(file_path):
    result = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for i, datum in enumerate(data):
        if i%2 == 0:
            result.append(HumanMessage(content=datum))
        else:
            result.append(AIMessage(content=datum))
    return result


def run_context():
    instructions = ""
    requirements = ""
    chat_history_boot = []
    if not os.path.exists(os.path.join(ROOT_FOLDER, 'agent')):
        os.mkdir(os.path.join(ROOT_FOLDER, 'agent'))
        with open(os.path.join(ROOT_FOLDER, 'agent', 'instructions.txt'), 'w', encoding='utf-8') as f:
            f.write("Напишите здесь описание проекта, и какую то дополнительную информацию")
        with open(os.path.join(ROOT_FOLDER, 'agent', 'requirements.txt'), 'w', encoding='utf-8') as f:
            f.write("Напишите здесь требования к коду, оформлению и другое.")
        with open(os.path.join(ROOT_FOLDER, 'agent', 'msg.json'), 'w', encoding='utf-8') as f:
            json.dump([], f)
    else:
        if os.path.exists(os.path.join(ROOT_FOLDER, 'agent', 'instructions.txt')):
            with open(os.path.join(ROOT_FOLDER, 'agent', 'instructions.txt'), 'r', encoding='utf-8') as f:
                instructions = f.read()
        else:
            with open(os.path.join(ROOT_FOLDER, 'agent', 'instructions.txt'), 'w', encoding='utf-8') as f:
                f.write("Напишите здесь описание проекта, и какую то дополнительную информацию")
        if os.path.exists(os.path.join(ROOT_FOLDER, 'agent', 'requirements.txt')):
            with open(os.path.join(ROOT_FOLDER, 'agent', 'requirements.txt'), 'r', encoding='utf-8') as f:
                requirements = f.read()
        else:
            with open(os.path.join(ROOT_FOLDER, 'agent', 'requirements.txt'), 'w', encoding='utf-8') as f:
                f.write("Напишите здесь требования к коду, оформлению и другое.")
        if os.path.exists(os.path.join(ROOT_FOLDER, 'agent', 'msg.json')):
            chat_history_boot = list_from_msg(os.path.join(ROOT_FOLDER, 'agent', 'msg.json'))
        else:
            with open(os.path.join(ROOT_FOLDER, 'agent', 'msg.json'), 'w', encoding='utf-8') as f:
                json.dump([], f)
    return {'instructions':instructions, 'requirements':requirements, 'chat_history':chat_history_boot}


if __name__ == "__main__":
    input_message = input('Введите ваше сообщение:\n')
    while input_message != '':
        data = run_context()
        chat_history = data['chat_history']
        instructions = data['instructions']
        requirements = data['requirements']
        input_string = f"""Запрос пользователя:{input_message}
        Специальные требования к коду: {requirements}
        Дополнительные инструкции: {instructions}
        """
        
        result = agent_executor.invoke({'input':input_string, 'chat_history':chat_history})['output']
        chat_history.extend(
            [
                HumanMessage(content=input_message),
                AIMessage(content=result)
            ]
        )
        if len(chat_history) > 10:
            chat_history = chat_history[2:]
        msg_from_list(chat_history=chat_history, file_path=os.path.join(ROOT_FOLDER, 'agent', 'msg.json'))
        input_message = input('Введите ваше сообщение:\n')

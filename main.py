from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain.pydantic_v1 import BaseModel, Field
load_dotenv()

ROOT_FOLDER = input("Введите абсолютный путь к корневому каталогу проекта.")

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY")
os.environ['LANGCHAIN_PROJECT'] = "Agent Strike"
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")

llm = ChatOpenAI(
        temperature=0,
        model="anthropic/claude-3.5-sonnet",#openai/gpt-4o
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
    )



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

@tool("read-file", args_schema=ReadInput, return_direct=True)
def read_file(file_path:str) -> str:
    """Инструмент для получения содержимого файла

    Args:
        file_path (str): Относительный путь к файлу в рабочей папке пользователя. Например folder/file.py или main.py. Полный список файлов и их местонахождения можно получить с помощью другого инструмента. Обязательно сначала удостоверься что файл существует.
    Returns:
        str: Содержимое файла либо сообщение что файла не существует.
    """
    if not os.path.exists(os.path.join(ROOT_FOLDER, file_path)):
        return "Файла не существует"
    with open(os.path.join(ROOT_FOLDER, file_path), 'r', encoding='utf-8') as f:
        return '\n'.join(f.readlines())

#Инструмент перезаписи

class WriteInput(BaseModel):
    file_path: str = Field(description="Относительный путь к файлу в рабочей папке пользователя. Например folder/file.py или main.py. Если файла не существует, для записи создастся новый файл с нужным именем. Если файл существует ты перезапишешь его содержимое. Полный список файлов и их местонахождения можно получить с помощью другого инструмента.")
    content: str = Field(description="Должно быть контентом для перезаписи или записи файла. Многострочный программный код или другое наполнение.")


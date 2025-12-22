import os
import json
from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
import openai
import logging

# Создаём класс для CLI-бота
class CliBot():

    def __init__(self, model_name):
        # Создаём модель
        self.chat_model = ChatOpenAI(
            model_name=model_name,
            temperature=0,
            max_tokens=300,
            request_timeout=15
        )

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Создаём Хранилище истории
        self.store = {}

        with open(r'data\faq.json', 'r') as f:
            self.faq = json.load(f)

        with open(r'data\orders.json', 'r') as f:
            self.orders = json.load(f)

        self.system_prompt = f"""Ты чат-бот поддержки. Отвечай вежливо и кратко и по делу.
        Информация о FAQ: {[[f'Вопрос {i["q"]}', f'Ответ {i["a"]}'] for i in self.faq]}
        Информация о заказах: {[[f'Заказ {q}', f'Статус {list(a.items())}'] for q,a in self.orders.items()]}
        """
        print(self.system_prompt)

        # Создаем шаблон промпта
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        # Создаём цепочку (тут используется синтаксис LCEL*)
        self.chain = self.prompt | self.chat_model

        # Создаём цепочку с историей
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,  # Цепочка с историей
            self.get_session_history,  # метод для получения истории
            input_messages_key="question",  # ключ для вопроса
            history_messages_key="history",  # ключ для истории
        )

    def setup_log(self, log_file):

        from pathlib import Path
        log_path = Path(log_file)

        log_path.parent.mkdir(parents=True, exist_ok=True)

        for handler in self.logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                self.logger.removeHandler(handler)
                handler.close()

        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        # Добавляем обработчик к логгеру
        self.logger.addHandler(file_handler)

    def trim_history(self, session_id):

        messages = self.store.get(session_id, [])

        if len(messages) <= 3:
            return None

        return {
            'messages': [messages[0]] + messages[-3:]
        }

    # Метод для получения истории по session_id
    def get_session_history(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def __call__(self, session_id):
        print("Чат-бот запущен! Можете задавать вопросы. \n - Для выхода введите 'выход'.\n - Для очистки контекста введите 'сброс'.\n")
        self.setup_log(f'logs\\session_{session_id}.log')

        while True:
            try:
                user_text = input("Вы: ").strip()
                self.logger.info(f"User: {user_text}")
            except (KeyboardInterrupt, EOFError):
                print("\nБот: Завершение работы.")
                break
            if not user_text:
                continue

            msg = user_text.lower()
            if msg in ("выход", "стоп", "конец"):
                print("Бот: До свидания!")
                break

            try:
                responce = self.chain_with_history.invoke(
                    {"question": user_text},
                    {"configurable": {"session_id": session_id}}
                )
                bot_reply = responce.content.strip()
                token_cnt = responce.usage_metadata['total_tokens']
                print(responce.usage_metadata)
                log_msg = json.dumps({'q': user_text, 'a': bot_reply, 't': token_cnt}, ensure_ascii=False)
                self.logger.info(log_msg)
                print('Бот:', bot_reply)
            except openai.APITimeoutError as e:
                print("Бот: [Ошибка] Превышено время ожидания ответа.")
                self.logger.error(json.dumps({'q': user_text, 'e': e}, ensure_ascii=False))
                continue
            except openai.APIConnectionError as e:
                print("Бот: [Ошибка] Не удалось подключиться к сервису LLM.")
                self.logger.error(json.dumps({'q': user_text, 'e': e}, ensure_ascii=False))
                continue
            except openai.AuthenticationError as e:
                print("Бот: [Ошибка] Проблема с API‑ключом (неавторизовано).")
                self.logger.error(json.dumps({'q': user_text, 'e': e}, ensure_ascii=False))
                break  # здесь можно завершить, т.к. дальнейшая работа бессмысленна
            except Exception as e:
                print(f"Бот: [Неизвестная ошибка] {e}")
                self.logger.error(json.dumps({'q': user_text, 'e': e}, ensure_ascii=False))
                continue


if __name__ == "__main__":
    model = os.getenv("OPENAI_API_MODEL", "gpt-4o-mini")

    bot = CliBot(
        model_name=model
    )
    bot("user_123")
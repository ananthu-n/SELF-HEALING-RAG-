from app.llm.service import LLMService


def main():

    llm = LLMService()

    response = llm.complete(

        system_prompt="You are a helpful assistant.",

        user_prompt="What does RAG stand for?",

    )

    print()

    print(response.answer)


if __name__ == "__main__":

    main()
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
from typing import Dict, Any, List
import logging
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from config.settings import settings
from utils.vector_store import VectorStore
from utils.web_search import WebSearch
import tiktoken

logger = logging.getLogger(__name__)

class TaxationChatbot:
    def __init__(self):
        """
        세무 챗봇 초기화 메서드.
        필요한 모든 유틸리티 객체와 설정을 초기화합니다.
        """

        self.llm = ChatOpenAI(temperature=settings.TEMPERATURE)
        self.memory = ConversationBufferWindowMemory(k=5)
        self.conversation = ConversationChain(llm=self.llm, memory=self.memory)
        self.vector_store = VectorStore()
        self.web_search = WebSearch()
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.max_tokens = 4000

        self.vector_store.use_custom_collection(settings.COLLECTION_NAME2)
    
    async def get_response(self, business_id: int, user_input: str) -> str :
        
        # 1. VectorDB에서 businessId에 해당하는 모든 데이터 검색
        user_data = {}

        for partition in ['TransactionFile', 'IncomeTaxFile', 'Question', 'BusinessInfo']:
            results = self.vector_store.search_in_partition(user_input, partition_name=partition, business_id=business_id)
            user_data[partition] = results


         # 2. 세액 정보 검색 (businessId 없는 데이터)
        self.vector_store.use_custom_collection(settings.COLLECTION_NAME2)
        tax_info = self.vector_store.search_with_similarity_without_url(user_input, k=5, threshold=0.3)

        # 3. 다른 모든 컬렉션에서 businessId 없는 데이터 검색
        other_collections = ['VATInfo', 'IncomeRates', 'SimpleTransaction', 'TaxFlow', 'TaxMethod', 'TaxFiles']
        for collection in other_collections:
            results = self.vector_store.search_in_partition(user_input, partition_name=collection, k=5)
            user_data[collection] = results

        # 3. 모든 관련 정보 결합
        all_relevant_data = []
        for partition, data in user_data.items():
            all_relevant_data.extend(data)
        all_relevant_data.extend(tax_info)
        all_relevant_data.sort(key=lambda x: x.get('similarity', 0), reverse=True)

        logger.info(f"Retrieved data: {json.dumps(all_relevant_data, indent=2, ensure_ascii=False)}")


        # 컨텍스트 준비 및 응답 생성
        context = self._prepare_context(all_relevant_data)
        response = self._generate_response(user_input, context)

        # if all_relevant_data:
        #     response = self._generate_response(user_input, all_relevant_data)
        # else:
        #     web_results = await self.web_search.search(user_input)
        #     response = self._generate_response(user_input, web_results)

        # 대화 기록 저장
        self.save_to_memory(user_input, response)

        return response
    
    
    def _prepare_context(self, data :Any) -> str:
        """
        데이터를 기반으로 컨텍스트를 준비합니다.
        데이터가 문자열인 경우 그대로 반환합니다.
        """

        if isinstance(data, str):
            return data

        context_parts = []
        total_tokens = 0

        for item in data:
            try:
                content = json.loads(item['content'])
                # JSON 객체가 리스트 형태로 올 경우 리스트 아이템 각각을 처리
                if isinstance(content, list):
                    content_parsed = "\n".join([self._format_content(c) for c in content])
                elif isinstance(content, dict):
                    content_parsed = self._format_content(content)
                else:
                    content_parsed = str(content)
            except json.JSONDecodeError:
                content_parsed = item['content']

            context_part = f"[{item.get('partition', '알 수 없음')}]\n{content_parsed}"
            tokens = len(self.tokenizer.encode(context_part))

            if total_tokens + tokens > self.max_tokens:
                break

            context_parts.append(context_part)
            total_tokens += tokens
        
        return "\n\n".join(context_parts)
    
    def _format_content(self, content : Any) -> str:
        """
        콘텐츠를 포맷하여 문자열로 반환.
        """
        return '\n'.join([f"{k}: {v}" for k, v in content.items()])

            
    def _generate_response(self, user_input: str, context_data: str) -> str:
        context = self._prepare_context(context_data)
        prompt = f"""당신은 유능한 세무사입니다. 다음 정보를 참고하여 사용자의 질문에 답하세요.
        필요한 경우 추가 정보를 제공하고, 구체적인 단계나 예시를 들어 설명해주세요.
        정보가 부족하거나 없는 경우, 일반적인 지식을 활용하여 최선을 다해 어린아이도 알 수 있도록 쉽게 최대한 자세하게 답변해 주세요.
        계산식이 필요할 경우 계산식으로 알기 쉽게 표현하세요. 정확한 정보에 맞는 계산식을 사용해주세요.
        어린아이가 바로 이해할 수 없을 정도의 어려운 단어는 쉽게 풀어서 설명해주세요. 혹은 주어진 답변 하단에 추가적으로 해석을 달아주세요.

        정확한 정보를 필요로 하는 정보라면, 최대한 사용자가 미리 제공한 추가정보와 그에 맞는 정보를 찾아서 제공하세요.

        추가적인 정보가 필요하다면 사용자에게 추가적인 정보를 물어보도록 
        출처(url)가 있다면 정보 하단에 출처를 남겨주세요. (예시 : 출처 : http://example.com)

        묻는 질문에 대해서만 답변해주세요.

        참고 정보:
        {context_data}

        사용자: {user_input}
        AI 조수:"""

        logger.info(f"Prepared context : {context_data}")
        logger.info(f"Generated prompt: {prompt}")
    
        tokenizer = tiktoken.encoding_for_model("gpt-4")
        encoded_prompt = self.tokenizer.encode(prompt)
        if len(encoded_prompt) > self.max_tokens:
            truncated_prompt = tokenizer.decode(encoded_prompt[:self.max_tokens])
            prompt = f"{truncated_prompt}\n\nAI 조수:"

        response = self.conversation.predict(input=prompt)
        return response
            
        
    def save_to_memory(self, user_input: str, response: str) :
        self.memory.save_context({"input": user_input}, {"output" : response})

    async def run(self):
        print("챗봇이 시작되었습니다. '종료'를 입력하면 대화를 마칩니다.")
        while True:
            user_input = input("사용자 : ")
            if user_input.lower() == '종료':
                break
            response = await self.get_response(business_id=12345, user_input=user_input)
            print(f"챗봇 : {response}")
            self.save_to_memory(user_input, response)

async def main():
    chatbot = TaxationChatbot()
    business_id = 12345

    while True:
        user_input = input("질문을 입력하세요 (종료하려면 'quit'입력)")
        if user_input.lower() == 'quit' :
            break

        response = await chatbot.get_response(business_id, user_input)
        print(f"챗봇 : {response}")


if __name__ == "__main__":
    asyncio.run(main())


 
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
        prompt = f"""다음 정보를 참고하여 사용자의 질문에 답하세요.
        필요한 경우 추가 정보를 제공하고, 구체적인 단계나 예시를 들어 설명해주세요.
        정보가 부족하거나 없는 경우, 일반적인 지식을 활용하여 최선을 다해 답변해 주세요.
        
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


    #     self.llm = ChatOpenAI(temperature=settings.TEMPERATURE, api_key=settings.OPENAI_API_KEY)
    #     self.memory = ConversationBufferWindowMemory(k=5)
    #     self.short_term_memory = deque(maxlen=5)
    #     self.conversation = ConversationChain(llm=self.llm, memory=self.memory, verbose=True)
    #     self.vector_store = VectorStore()
    #     self.intent_analyzer = IntentAnalyzer()
    #     self.query_generator = QueryGenerator()
    #     self.web_search = WebSearch()
    #     self.max_tokens = 14000
    #     self.settings = settings
    
    #      # 세무 전용 벡터 저장소 사용
    #     self.vector_store.use_custom_collection(settings.COLLECTION_NAME2)

    # async def get_taxation_response(self, business_id: int, user_input: str) -> str:
    #     """
    #     세무 관련 질문에 대한 응답을 생성합니다.
    #     """
    #     try:
    #         # 질문 의도 분석
    #         intent = self.intent_analyzer.analyze_intent(user_input)
    #         logger.info(f"분석된 의도: {intent}")

    #         # 세무 관련 데이터 검색
    #         taxation_data = self.get_taxation_data(business_id)
    #         relevant_info = self._get_taxation_relevant_info(user_input, intent)

    #         # 세무 데이터를 적절히 추가
    #         relevant_info.extend(taxation_data)

    #         # 문맥 생성 및 프롬프트 준비
    #         context = self._prepare_context(relevant_info)
    #         prompt = self._create_prompt(user_input, context)

    #         # 응답 생성
    #         response = self.conversation.predict(input=prompt)
    #         logger.info("응답이 성공적으로 생성되었습니다.")
    #         return response

    #     except Exception as e:
    #         logger.error(f"응답 생성 중 오류 발생: {str(e)}")
    #         return "요청을 처리하는 동안 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."

    # def get_taxation_data(self, business_id: int) -> List[Dict[str, Any]]:
    #     """
    #     특정 사업자 ID에 해당하는 세무 데이터를 벡터 저장소에서 가져옵니다.
    #     """
    #     return self.vector_store.get_data_by_business_id(business_id)

    # def _get_taxation_relevant_info(self, user_input: str, intent: str) -> List[Dict[str, Any]]:
    #     """
    #     세무 관련 정보를 벡터 저장소 또는 웹 검색을 통해 가져옵니다.
    #     """
    #     relevant_info = self.vector_store.search_with_similarity_without_url(user_input, k=20, threshold=0.7)
    #     if not relevant_info:
    #         generated_queries = self.query_generator.generate_queries(user_input, intent)
    #         web_results = self.web_search.search(generated_queries)
    #         relevant_results = [
    #             result for result in web_results 
    #             if self._is_relevant(result['content'], user_input)
    #         ]
    #         if relevant_results:
    #             self.vector_store.add_search_results(relevant_results)
    #             return relevant_results
        
    #     return relevant_info or [{"content": "요청하신 정보에 대한 구체적인 데이터를 찾지 못했습니다.", "url": ""}]

    # def _prepare_context(self, relevant_info: List[Dict[str, Any]]) -> str:
    #     """
    #     검색 결과와 단기 기억을 기반으로 문맥을 준비합니다.
    #     """
    #     context_parts = []
    #     for info in relevant_info:
    #         part = f"[제목: {info['title']}]\n{info['snippet']}"
    #         if info.get('url'):
    #             part += f"\n[출처: {info['url']}]"
    #         context_parts.append(part)
        
    #     return "\n\n".join(context_parts) if context_parts else "관련된 구체적인 정보를 찾지 못했습니다."

    # def _is_relevant(self, content: str, user_input: str) -> bool:
    #     """
    #     웹 검색 결과가 사용자 입력과 관련 있는지 판단합니다.
    #     """
    #     # 간단한 키워드 매칭이나 유사도 분석으로 관련성을 판단
    #     return user_input in content

    # def _create_prompt(self, user_input: str, context: str) -> str:
    #     """
    #     사용자 입력과 문맥을 바탕으로 프롬프트를 생성합니다.
    #     """
    #     return f"""다음은 사용자의 요청사항에 관련된 정보입니다. 이 정보를 바탕으로 다음과 같이 응답해주세요:

    #     1. 관련 정보를 간결하게 정리해주시고, 간단한 설명을 제공해주세요.
    #     2. 구체적인 예시, 수치, 날짜 등을 포함해주세요.
    #     3. 정보가 부족한 경우 그 사실을 언급해주세요.
    #     4. 사용자가 더 궁금해 할 만한 사항에 대해 추가 질문을 추천해주세요.

    #     참고 정보:
    #     {context}

    #     사용자: {user_input}
    #     AI 조수:"""

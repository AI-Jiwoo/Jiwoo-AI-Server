import requests
from hashlib import md5
from services.taxation import vector_store


def download_image(url):
    """
    주어진 URL에서 이미지를 다운로드하고 바이트 데이터를 반환
    """
    response = requests.get(url)
    response.raise_for_status()  # 요청이 성공적으로 완료되었는지 확인
    return response.content

def calculate_md5(data):
    """
    주어진 데이터의 MD5 해시값을 계산하여 반환
    """
    return md5(data).hexdigest()

def store_image_and_url(url, metadata):
    """
    이미지를 VectorDB에 저장하고 URL도 함께 저장
    """
    image_data = download_image(url)
    new_md5 = calculate_md5(image_data)
    
    # 이미지와 URL을 결합한 텍스트로 VectorDB에 저장
    combined_text = f"{metadata}\nURL: {url}\nMD5: {new_md5}\n"
    vector_store.add_texts([combined_text])
    print(f"이미지 및 URL 저장 완료: {metadata['type']}")

def store_url_content(url, metadata):
    """
    주어진 URL에서 콘텐츠를 가져와 VectorDB에 저장
    """
    response = requests.get(url)
    response.raise_for_status() # 요청이 성공적으로 완료되었는지 확인
    text_content = response.text.strip()

    combined_text = f"{metadata}\n{text_content}"
    vector_store.add_texts([combined_text])
    print(f"URL 콘텐츠 저장 완료: {metadata['type']}")

def store_tax_calculation_data():
    """
    세액 계산 흐름도 이미지와 URL, 그리고 세액 계산 방법을 VectorDB에 저장
    """

    try: 
        # 세액 계산 흐름도 이미지와 URL
        flow_url = "https://www.nts.go.kr/nts/cm/cntnts/cntntsView.do?mi=2226&cntntsId=7666"
        flow_metadata = {"type": "tax_calculation_flow_image", "source": flow_url}
        store_image_and_url(flow_url, flow_metadata)
        
        # 세액 계산 방법 URL
        method_url = "https://www.nts.go.kr/nts/cm/cntnts/cntntsView.do?mi=6591&cntntsId=7870"
        method_metadata = {"type": "tax_calculation_method", "source": method_url}
        store_url_content(method_url, method_metadata)

        # 세액공제 
        

    except requests.exceptions.RequestException as e:
        print(f"데이터 저장 중 오류 발생: {e}")
    except Exception as e:
        print(f"예상치 못한 오류 발생: {e}")

# 함수 호출하여 저장 또는 업데이트 수행
store_tax_calculation_data()

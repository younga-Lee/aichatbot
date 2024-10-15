import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import json
import re
import os

app = FastAPI()

#api key가져오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")

#챗봇 사용자 요구사항
class MenuRequest(BaseModel):
    customer_diagnosis: str
    menu_count: int

#챗봇 출력사항 
class MenuItem(BaseModel):
    product_name: str
    description: str
    recipe: str
    ingredients_cost: float
    price: float
    image_url: str

@app.post("/generate-menu/", response_model=List[MenuItem])
async def generate_menu(request: MenuRequest):
    if request.menu_count < 1 or request.menu_count > 3:
        raise HTTPException(status_code=400, detail="menu_count must be between 1 and 3")
    
    prompt = f"""
    고객 진단 결과: {request.customer_diagnosis}

    메뉴 {request.menu_count}개를 JSON 형식으로 제안해 주세요. 형식은 다음과 같습니다:
    [
        {{
            "product_name": "제품 이름",
            "description": "설명",
            "recipe": "레시피",
            "ingredients_cost": 재료비,
            "price": 판매가격,
            "image_url": "그림 URL"
        }}
    ]
    제안:
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        )
        generated_text = response['choices'][0]['message']['content'].strip()
        
        # JSON 형식의 응답을 파싱
        try:
            menu_items = json.loads(generated_text)  # JSON 형식으로 변환
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"응답 파싱 중 오류 발생: {str(e)}")
        
        # 각 메뉴 항목에 대해 이미지 생성
        for item in menu_items:
            image_prompt = f"{item['product_name']} - {item['description']}"
            try:
                image_response = openai.Image.create(
                    prompt=image_prompt,
                    n=1,
                    size="512x512"
                )
                item['image_url'] = image_response['data'][0]['url']
            except openai.error.OpenAIError as e:
                item['image_url'] = "이미지 생성 실패"  # 이미지 생성에 실패할 경우 기본값 설정

        return menu_items
        
    except openai.error.OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API Error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected Error: {str(e)}")